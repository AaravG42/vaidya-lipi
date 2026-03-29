from __future__ import annotations
import base64, json, logging, os, sys, uuid
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import gradio as gr
import numpy as np

# MUST come before any other gradio use — patches the schema crash
import gradio_client.utils as _gc_utils
_orig_inner = _gc_utils._json_schema_to_python_type
_orig_get_type = _gc_utils.get_type

def _safe_inner(schema, defs=None):
    if not isinstance(schema, dict):
        return "Any"
    return _orig_inner(schema, defs)

def _safe_get_type(schema):
    if not isinstance(schema, dict):
        return "Any"
    return _orig_get_type(schema)

_gc_utils._json_schema_to_python_type = _safe_inner
_gc_utils.get_type = _safe_get_type


def _load_secrets():
    mapping = {
        "SARVAM_API_KEY":  ("vaidya-lipi", "sarvam_api_key"),
        "HF_TOKEN":        ("vaidya-lipi", "hf_token"),
    }
    for env_var, (scope, key) in mapping.items():
        if os.environ.get(env_var, "").strip():
            continue
        try:
            from databricks.sdk import WorkspaceClient
            w = WorkspaceClient()
            val = w.secrets.get_secret(scope=scope, key=key)
            if val and val.value:
                try:
                    decoded = base64.b64decode(val.value).decode("utf-8")
                except Exception:
                    decoded = val.value
                os.environ[env_var] = decoded
        except Exception as e:
            logging.warning("Could not load %s: %s", env_var, e)


def transcribe_audio(audio_numpy) -> tuple[str, str]:
    import io, wave, requests

    if audio_numpy is None:
        return "", "en-IN"

    sr, data = audio_numpy
    data = np.asarray(data)

    # Convert to mono if stereo
    if data.ndim == 2:
        data = data.mean(axis=1)

    # Convert to 16-bit PCM WAV
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        if data.dtype != np.int16:
            data = (data * 32767).clip(-32768, 32767).astype(np.int16)
        wf.writeframes(data.tobytes())
    wav_bytes = buf.getvalue()

    api_key = os.environ.get("SARVAM_API_KEY", "")
    if not api_key:
        raise ValueError("SARVAM_API_KEY not set")

    response = requests.post(
        "https://api.sarvam.ai/speech-to-text",
        headers={"api-subscription-key": api_key},   # correct header name
        files={"file": ("audio.wav", wav_bytes, "audio/wav")},
        data={
            "model": "saaras:v3",           # correct model name
            "mode": "transcribe",           # transcribe keeps original language
            "language_code": "unknown",     # auto-detect
        },
        timeout=30,
    )

    if response.status_code != 200:
        raise ValueError(f"Sarvam API error {response.status_code}: {response.text}")

    result = response.json()
    transcript = result.get("transcript", "")
    detected_lang = result.get("language_code", "en-IN")
    return transcript, detected_lang

# Load model once at startup — this is an embedding model, fine on CPU
_parrotlet_model = None
_parrotlet_tokenizer = None
_faiss_index = None
_faiss_metadata = None

def _load_parrotlet():
    global _parrotlet_model, _parrotlet_tokenizer, _faiss_index, _faiss_metadata
    if _parrotlet_model is not None:
        return

    import torch, faiss
    from transformers import AutoTokenizer, AutoModel

    hf_token = os.environ.get("HF_TOKEN", "")
    model_name = "ekacare/parrotlet-e"
    _parrotlet_tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    _parrotlet_model = AutoModel.from_pretrained(model_name, token=hf_token)
    _parrotlet_model.eval()

    # Load FAISS index — download from UC Volume to /tmp
    index_dir = "/tmp/parrotlet_index"
    os.makedirs(index_dir, exist_ok=True)
    index_path = f"{index_dir}/index.faiss"
    meta_path = f"{index_dir}/metadata.json"

    if not os.path.exists(index_path):
        try:
            from databricks.sdk import WorkspaceClient
            w = WorkspaceClient()
            vol = "/Volumes/workspace/vaidya/models_and_indexes/parrotlet_index"
            for fname in ["index.faiss", "metadata.json"]:
                with w.files.download(f"{vol}/{fname}").contents as src:
                    with open(f"{index_dir}/{fname}", "wb") as dst:
                        dst.write(src.read())
        except Exception as e:
            logging.warning("Could not download FAISS index: %s", e)
            return

    _faiss_index = faiss.read_index(index_path)
    with open(meta_path) as f:
        _faiss_metadata = json.load(f)
    logging.info("Parrotlet-e + FAISS index loaded: %d vectors", _faiss_index.ntotal)

def extract_medical_entities(text: str, top_k: int = 5) -> list[dict]:
    """Map free text to SNOMED concepts using Parrotlet-e embeddings."""
    import torch
    _load_parrotlet()
    if _parrotlet_model is None or _faiss_index is None:
        return []

    encoded = _parrotlet_tokenizer(
        [text], padding=True, truncation=True, max_length=128, return_tensors="pt"
    )
    with torch.no_grad():
        output = _parrotlet_model(**encoded)
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        emb = (output.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        emb = torch.nn.functional.normalize(emb, p=2, dim=1).numpy()

    scores, indices = _faiss_index.search(emb, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0 and float(score) > 0.5:  # confidence threshold
            meta = _faiss_metadata[idx]
            results.append({
                "term": meta["term"],       # English SNOMED term
                "concept_id": meta["concept_id"],
                "score": float(score),
            })
    return results


SOAP_PROMPT = """You are a medical scribe AI for Indian hospitals (ABDM-compliant system).
Given the transcript of a doctor-patient consultation, extract and structure it into:
1. SYMPTOMS: List of symptoms the patient described
2. MEDICATIONS: Any medications mentioned (dosage if stated)
3. DIAGNOSIS: Doctor's assessment or working diagnosis
4. PLAN: Follow-up, tests ordered, advice given

Also provide a full SOAP note:
- S (Subjective): Patient's complaints in their own words
- O (Objective): Any clinical findings mentioned
- A (Assessment): Diagnosis / differential
- P (Plan): Treatment plan

Respond ONLY as valid JSON with keys:
symptoms, medications, diagnosis, plan, soap_s, soap_o, soap_a, soap_p

Transcript:
"""

# def structure_transcript(transcript: str) -> dict:
#     """Call Llama 4 Maverick via AI Gateway to structure the transcript."""
#     import requests

#     base_url = os.environ.get("LLM_OPENAI_BASE_URL", "")
#     model = os.environ.get("LLM_MODEL", "databricks-llama-4-maverick")

#     if not base_url:
#         raise ValueError("LLM_OPENAI_BASE_URL not set in app.yaml")

#     # Get OAuth token (works on Databricks Apps, no PAT needed)
#     from databricks.sdk import WorkspaceClient
#     w = WorkspaceClient()
#     headers_auth = w.config.authenticate()

#     payload = {
#         "model": model,
#         "messages": [
#             {"role": "system", "content": SOAP_PROMPT},
#             {"role": "user", "content": transcript}
#         ],
#         "max_tokens": 1024,
#         "temperature": 0.1,  # low temp for structured extraction
#     }

#     resp = requests.post(
#         f"{base_url}/chat/completions",
#         headers={**headers_auth, "Content-Type": "application/json"},
#         json=payload,
#         timeout=30,
#     )
#     if not resp.ok:
#         print("AI Gateway 400 body:", resp.text)   # ADD THIS
#         resp.raise_for_status()
#     resp.raise_for_status()
#     content = resp.json()["choices"][0]["message"]["content"]

#     # Strip markdown fences if present
#     content = content.strip()
#     if content.startswith("```"):
#         content = content.split("```")[1]
#         if content.startswith("json"):
#             content = content[4:]
#     content = content.strip()

#     try:
#         return json.loads(content)
#     except json.JSONDecodeError:
#         # Fallback: return raw text in a structured wrapper
#         return {
#             "symptoms": [], "medications": [], "diagnosis": "See raw note",
#             "plan": content, "soap_s": transcript,
#             "soap_o": "", "soap_a": "", "soap_p": ""
#         }
def structure_transcript(transcript: str) -> dict:
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

    w = WorkspaceClient()

    messages = [
        ChatMessage(role=ChatMessageRole.SYSTEM, content=SOAP_PROMPT),
        ChatMessage(role=ChatMessageRole.USER,   content=transcript),
    ]

    response = w.serving_endpoints.query(
        name="databricks-meta-llama-3-3-70b-instruct",
        messages=messages,
        max_tokens=1024,
        temperature=0.1,
    )

    # Response is a QueryEndpointResponse object — use as_dict()
    response_dict = response.as_dict()
    content = response_dict["choices"][0]["message"]["content"].strip()

    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {
            "symptoms": [], "medications": [], "diagnosis": "See raw note",
            "plan": content, "soap_s": transcript,
            "soap_o": "", "soap_a": "", "soap_p": ""
        }

def _get_sql_connection():
    """Get a SQL warehouse connection — works inside Databricks Apps."""
    from databricks import sql
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    host = w.config.host.replace("https://", "")

    # Find the first running SQL warehouse
    warehouses = list(w.warehouses.list())
    running = [wh for wh in warehouses if str(wh.state).upper() == "RUNNING"]
    warehouse_id = running[0].id if running else warehouses[0].id

    token = w.config.authenticate().get("Authorization", "").replace("Bearer ", "")

    return sql.connect(
        server_hostname=host,
        http_path=f"/sql/1.0/warehouses/{warehouse_id}",
        access_token=token,
    )


def save_record(patient_id: str, doctor_id: str, transcript: str,
                structured: dict, entities: list, language: str) -> str:
    record_id = str(uuid.uuid4())
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    symptoms    = json.dumps(structured.get("symptoms", []))
    medications = json.dumps(structured.get("medications", []))
    snomed_codes = json.dumps([e["concept_id"] for e in entities])
    structured_note = json.dumps(structured)

    # Escape single quotes in text fields
    def esc(s): return (s or "").replace("'", "''")

    sql_stmt = f"""
    INSERT INTO workspace.vaidya.patient_records
        (record_id, patient_id, doctor_id, hospital_id, timestamp,
         language_detected, raw_transcript, structured_note,
         soap_subjective, soap_objective, soap_assessment, soap_plan,
         is_anonymized)
    VALUES
        ('{record_id}', '{esc(patient_id)}', '{esc(doctor_id)}',
         '{os.environ.get("HOSPITAL_ID","DEMO_HOSPITAL")}', '{now}',
         '{language}', '{esc(transcript)}', '{esc(structured_note)}',
         '{esc(structured.get("soap_s",""))}',
         '{esc(structured.get("soap_o",""))}',
         '{esc(structured.get("soap_a",""))}',
         '{esc(structured.get("soap_p",""))}',
         false)
    """

    with _get_sql_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql_stmt)

    return record_id


def get_doctor_dashboard(doctor_id: str) -> dict:
    query = f"""
        SELECT
            COUNT(*) as total,
            language_detected
        FROM workspace.vaidya.patient_records
        WHERE doctor_id = '{doctor_id}'
        AND DATE(timestamp) = CURRENT_DATE()
        GROUP BY language_detected
    """

    symptom_query = f"""
    SELECT symptom, COUNT(*) as cnt FROM (
        SELECT explode(
            from_json(structured_note, 'struct<symptoms:array<string>>').symptoms
        ) as symptom
        FROM workspace.vaidya.patient_records
        WHERE doctor_id = '{doctor_id}'
        AND DATE(timestamp) = CURRENT_DATE()
    )
    WHERE symptom IS NOT NULL
    GROUP BY symptom ORDER BY cnt DESC LIMIT 5
    """

    with _get_sql_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            total = sum(r[0] for r in rows)
            languages = {r[1]: r[0] for r in rows}

            cursor.execute(symptom_query)
            sym_rows = cursor.fetchall()
            top_symptoms = [(r[0], r[1]) for r in sym_rows]

    return {
        "total_patients_today": total,
        "top_symptoms": top_symptoms,
        "languages": languages,
    }

_CSS = """
/* ── Dark Mode Tokens ─────────────────────────────────────── */
:root {
  --vl-bg:          #0B1120;
  --vl-surface:     #131F35;
  --vl-surface-hi:  #1A2A45;
  --vl-border:      #1E3050;
  --vl-border-hi:   #2A4268;
  --vl-teal:        #2DD4BF;
  --vl-teal-btn:    #0D9488;
  --vl-teal-btn-dk: #0F766E;
  --vl-teal-pale:   rgba(45,212,191,0.07);
  --vl-teal-ring:   rgba(45,212,191,0.2);
  --vl-blue:        #38BDF8;
  --vl-text:        #E2E8F0;
  --vl-text-muted:  #94A3B8;
  --vl-text-dim:    #475569;
  --vl-green:       #34D399;
  --vl-red:         #F87171;
  --vl-amber:       #FBBF24;
  --vl-radius:      12px;
  --vl-shadow:      0 2px 8px rgba(0,0,0,0.4), 0 1px 3px rgba(0,0,0,0.3);
}

/* ── Global Dark Base ─────────────────────────────────────── */
body, html {
  background: var(--vl-bg) !important;
  color-scheme: dark;
}
.gradio-container {
  background: var(--vl-bg) !important;
  max-width: 1080px !important;
  margin: 0 auto !important;
  padding: 0 !important;
  font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif !important;
  color: var(--vl-text) !important;
}
footer { display: none !important; }
.contain { padding: 0 !important; background: var(--vl-bg) !important; }

/* Gradio internal wrappers — force dark */
.block, .form, .gap, .padded,
.wrap:not(.vl-header *),
div[class*="gradio-"] {
  background: transparent !important;
}
.tabitem { background: transparent !important; }

/* ── App Header ───────────────────────────────────────────── */
.vl-header {
  background: linear-gradient(135deg, #0B4D45 0%, #0B3A5C 100%);
  border-radius: 0 0 20px 20px;
  padding: 26px 32px 30px;
  margin-bottom: 20px;
  border-bottom: 1px solid #1A4060;
}
.vl-header-inner {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 14px;
}
.vl-logo {
  display: flex;
  align-items: center;
  gap: 14px;
  margin-bottom: 4px;
}
.vl-logo-icon {
  width: 46px; height: 46px;
  background: rgba(45,212,191,0.15);
  border: 1px solid rgba(45,212,191,0.35);
  border-radius: 12px;
  display: flex; align-items: center; justify-content: center;
  font-size: 22px;
  flex-shrink: 0;
}
.vl-logo h1 {
  font-size: 1.75rem !important;
  font-weight: 800 !important;
  color: #E2E8F0 !important;
  margin: 0 !important;
  letter-spacing: -0.4px !important;
  line-height: 1.1 !important;
}
.vl-deva {
  font-size: 0.9rem;
  color: rgba(226,232,240,0.6);
  margin: 0;
  font-weight: 400;
}
.vl-tagline {
  font-size: 0.82rem;
  color: rgba(226,232,240,0.5);
  margin: 6px 0 0 0;
}
.vl-badges {
  display: flex;
  gap: 7px;
  align-items: center;
  flex-wrap: wrap;
  margin-top: 4px;
}
.vl-badge {
  background: rgba(45,212,191,0.12);
  border: 1px solid rgba(45,212,191,0.3);
  border-radius: 20px;
  padding: 3px 11px;
  font-size: 0.7rem;
  font-weight: 700;
  color: var(--vl-teal);
  letter-spacing: 0.8px;
  text-transform: uppercase;
}

/* ── Cards ────────────────────────────────────────────────── */
.vl-card {
  background: var(--vl-surface) !important;
  border-radius: var(--vl-radius) !important;
  box-shadow: var(--vl-shadow) !important;
  border: 1px solid var(--vl-border) !important;
  padding: 20px 22px !important;
  margin-bottom: 14px !important;
}
.vl-card > .wrap,
.vl-card > div > .wrap { border: none !important; background: transparent !important; }

/* ── Section Labels ───────────────────────────────────────── */
.vl-section-label {
  font-size: 0.7rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 1.1px;
  color: var(--vl-text-dim);
  margin: 0 0 14px 0;
  padding-bottom: 10px;
  border-bottom: 1px solid var(--vl-border);
}

/* ── Workflow Steps ───────────────────────────────────────── */
.vl-steps {
  display: flex;
  align-items: center;
  padding: 12px 4px 18px;
  flex-wrap: wrap;
  gap: 4px;
}
.vl-step {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 0.8rem;
  color: var(--vl-text-muted);
  font-weight: 500;
  white-space: nowrap;
}
.vl-step-num {
  width: 26px; height: 26px;
  border-radius: 50%;
  background: var(--vl-surface);
  color: var(--vl-text-muted);
  display: flex; align-items: center; justify-content: center;
  font-size: 0.72rem; font-weight: 700;
  flex-shrink: 0;
  border: 1.5px solid var(--vl-border-hi);
}
.vl-step-line {
  width: 36px; height: 2px;
  background: var(--vl-border);
  margin: 0 2px;
  flex-shrink: 0;
}

/* ── Tabs ─────────────────────────────────────────────────── */
.tabs > .tab-nav { border-bottom: 2px solid var(--vl-border) !important; background: transparent !important; }
.tabs > .tab-nav > button {
  font-weight: 600 !important;
  font-size: 0.88rem !important;
  color: var(--vl-text-muted) !important;
  padding: 10px 18px !important;
  border-radius: 0 !important;
  border-bottom: 2px solid transparent !important;
  margin-bottom: -2px !important;
  background: transparent !important;
}
.tabs > .tab-nav > button.selected {
  color: var(--vl-teal) !important;
  border-bottom-color: var(--vl-teal) !important;
  background: transparent !important;
}
.tabs > .tab-nav > button:hover:not(.selected) {
  color: var(--vl-text) !important;
  background: rgba(255,255,255,0.04) !important;
}

/* ── Form Labels ──────────────────────────────────────────── */
label > span,
.vl-card label > span:first-child {
  font-size: 0.75rem !important;
  font-weight: 600 !important;
  color: var(--vl-text-muted) !important;
  text-transform: uppercase !important;
  letter-spacing: 0.6px !important;
}

/* ── Form Inputs ──────────────────────────────────────────── */
input[type="text"],
input[type="number"],
textarea {
  border-radius: 8px !important;
  border-color: var(--vl-border-hi) !important;
  background: var(--vl-bg) !important;
  color: var(--vl-text) !important;
  font-size: 0.9rem !important;
  transition: border-color 0.15s, box-shadow 0.15s !important;
}
input[type="text"]:focus,
input[type="number"]:focus,
textarea:focus {
  border-color: var(--vl-teal) !important;
  box-shadow: 0 0 0 3px var(--vl-teal-ring) !important;
  outline: none !important;
}
input::placeholder, textarea::placeholder {
  color: var(--vl-text-dim) !important;
}

/* ── Buttons ──────────────────────────────────────────────── */
button.primary {
  background: var(--vl-teal-btn) !important;
  border-color: var(--vl-teal-btn) !important;
  color: white !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
  font-size: 0.88rem !important;
  padding: 9px 20px !important;
  transition: background 0.15s, transform 0.1s, box-shadow 0.15s !important;
}
button.primary:hover {
  background: var(--vl-teal-btn-dk) !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 16px rgba(13,148,136,0.35) !important;
}
button.secondary {
  border-radius: 8px !important;
  font-weight: 600 !important;
  font-size: 0.88rem !important;
  padding: 9px 20px !important;
  border-color: var(--vl-border-hi) !important;
  color: var(--vl-text-muted) !important;
  background: var(--vl-surface-hi) !important;
  transition: background 0.15s, border-color 0.15s, color 0.15s !important;
}
button.secondary:hover {
  background: var(--vl-border) !important;
  border-color: var(--vl-text-dim) !important;
  color: var(--vl-text) !important;
}

/* ── Audio Block ──────────────────────────────────────────── */
.vl-audio .wrap,
.vl-audio .audio-container {
  border-radius: 10px !important;
  border: 2px dashed rgba(45,212,191,0.3) !important;
  background: var(--vl-teal-pale) !important;
}

/* ── Status ───────────────────────────────────────────────── */
.vl-status label > span { display: none !important; }
.vl-status textarea {
  font-size: 0.84rem !important;
  border-radius: 8px !important;
  background: var(--vl-surface-hi) !important;
  color: var(--vl-text-muted) !important;
  font-family: 'SF Mono', 'Fira Code', monospace !important;
  border-color: var(--vl-border) !important;
}

/* ── JSON viewers ─────────────────────────────────────────── */
.json-holder, pre {
  background: var(--vl-bg) !important;
  color: var(--vl-text) !important;
  border-color: var(--vl-border) !important;
  border-radius: 8px !important;
}

/* ── Dashboard stat ───────────────────────────────────────── */
.vl-stat input[type="number"] {
  font-size: 2.8rem !important;
  font-weight: 800 !important;
  color: var(--vl-teal) !important;
  text-align: center !important;
  border: none !important;
  background: var(--vl-teal-pale) !important;
  border-radius: 10px !important;
  padding: 16px !important;
}

/* ── Misc Gradio overrides ────────────────────────────────── */
.svelte-1gfkn6j, .panel { background: transparent !important; }
p, span, li { color: var(--vl-text) !important; }

/* ── Responsive ───────────────────────────────────────────── */
@media (max-width: 768px) {
  .vl-header { padding: 18px 20px 22px; }
  .vl-logo h1 { font-size: 1.4rem !important; }
  .vl-steps { gap: 8px; }
  .vl-step-line { width: 18px; }
}
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(
        css=_CSS,
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.teal,
            secondary_hue=gr.themes.colors.blue,
            neutral_hue=gr.themes.colors.slate,
        ).set(
            body_background_fill="#0B1120",
            body_background_fill_dark="#0B1120",
            block_background_fill="#131F35",
            block_background_fill_dark="#131F35",
            block_border_color="#1E3050",
            block_border_color_dark="#1E3050",
            input_background_fill="#0B1120",
            input_background_fill_dark="#0B1120",
            input_border_color="#2A4268",
            input_border_color_dark="#2A4268",
            body_text_color="#E2E8F0",
            body_text_color_dark="#E2E8F0",
            body_text_color_subdued="#94A3B8",
            body_text_color_subdued_dark="#94A3B8",
            button_primary_background_fill="#0D9488",
            button_primary_background_fill_dark="#0D9488",
            button_primary_text_color="white",
            button_primary_text_color_dark="white",
            button_secondary_background_fill="#1A2A45",
            button_secondary_background_fill_dark="#1A2A45",
            button_secondary_border_color="#2A4268",
            button_secondary_border_color_dark="#2A4268",
            button_secondary_text_color="#94A3B8",
            button_secondary_text_color_dark="#94A3B8",
            border_color_primary="#1E3050",
            border_color_primary_dark="#1E3050",
            background_fill_primary="#0B1120",
            background_fill_primary_dark="#0B1120",
            background_fill_secondary="#131F35",
            background_fill_secondary_dark="#131F35",
            color_accent="#2DD4BF",
            color_accent_soft="rgba(45,212,191,0.12)",
            color_accent_soft_dark="rgba(45,212,191,0.12)",
            shadow_drop="0 2px 8px rgba(0,0,0,0.5)",
            shadow_drop_lg="0 4px 16px rgba(0,0,0,0.6)",
        ),
        title="Vaidya Lipi — Medical Scribe",
    ) as demo:

        gr.HTML("""
        <div class="vl-header">
          <div class="vl-header-inner">
            <div>
              <div class="vl-logo">
                <div class="vl-logo-icon">⚕</div>
                <div>
                  <h1>Vaidya Lipi</h1>
                  <p class="vl-deva">वैद्य लिपि</p>
                </div>
              </div>
              <p class="vl-tagline">AI Medical Scribe — multilingual, ABDM-compatible</p>
            </div>
            <div class="vl-badges">
              <span class="vl-badge">ABDM</span>
              <span class="vl-badge">SNOMED CT</span>
              <span class="vl-badge">Multilingual</span>
            </div>
          </div>
        </div>
        """)

        # State lives OUTSIDE tabs so both tabs can share it
        doctor_id_state = gr.State("DR001")

        with gr.Tabs():

            # ── Tab 1: Record Consultation ─────────────────────────
            with gr.Tab("🎙  Record Consultation"):

                with gr.Group(elem_classes="vl-card"):
                    gr.HTML('<div class="vl-section-label">Patient Information</div>')
                    with gr.Row():
                        patient_id_box = gr.Textbox(
                            label="Patient ID",
                            placeholder="ABHA ID or local ID  (e.g. PAT1234)",
                            scale=2,
                        )
                        doctor_id_box = gr.Textbox(
                            label="Doctor ID",
                            value="DR001",
                            scale=1,
                        )

                gr.HTML("""
                <div class="vl-steps">
                  <div class="vl-step">
                    <div class="vl-step-num">1</div>
                    <span>Enter patient details</span>
                  </div>
                  <div class="vl-step-line"></div>
                  <div class="vl-step">
                    <div class="vl-step-num">2</div>
                    <span>Record consultation</span>
                  </div>
                  <div class="vl-step-line"></div>
                  <div class="vl-step">
                    <div class="vl-step-num">3</div>
                    <span>Transcribe audio</span>
                  </div>
                  <div class="vl-step-line"></div>
                  <div class="vl-step">
                    <div class="vl-step-num">4</div>
                    <span>Structure &amp; save</span>
                  </div>
                </div>
                """)

                with gr.Group(elem_classes="vl-card vl-audio"):
                    gr.HTML('<div class="vl-section-label">Audio Recording</div>')
                    audio_input = gr.Audio(
                        sources=["microphone"],
                        type="numpy",
                        label="Record in Hindi, English, or any Indian language",
                    )

                with gr.Group(elem_classes="vl-card"):
                    gr.HTML('<div class="vl-section-label">Transcript</div>')
                    transcript_box = gr.Textbox(
                        label="",
                        lines=4,
                        placeholder="Transcript appears here after recording — or type manually…",
                        show_label=False,
                    )
                    with gr.Row():
                        transcribe_btn = gr.Button("↺  Transcribe Audio", variant="secondary")
                        process_btn = gr.Button("✦  Structure & Save Record", variant="primary")

                with gr.Row(equal_height=False):
                    with gr.Column():
                        with gr.Group(elem_classes="vl-card"):
                            gr.HTML('<div class="vl-section-label">SOAP Note</div>')
                            soap_box = gr.JSON(label="", show_label=False)
                    with gr.Column():
                        with gr.Group(elem_classes="vl-card"):
                            gr.HTML('<div class="vl-section-label">SNOMED Entities  (Parrotlet-e)</div>')
                            entities_box = gr.JSON(label="", show_label=False)

                status_box = gr.Textbox(
                    label="Status",
                    interactive=False,
                    elem_classes="vl-status",
                )

                # ── Sync doctor ID into shared State ──────────────
                doctor_id_box.change(
                    fn=lambda x: x,
                    inputs=[doctor_id_box],
                    outputs=[doctor_id_state],
                )

                def on_transcribe(audio):
                    if audio is None:
                        return "No audio recorded. Please use the microphone."
                    try:
                        transcript, lang = transcribe_audio(audio)
                        return f"[{lang}] {transcript}"
                    except Exception as e:
                        return f"Transcription error: {e}"

                def on_process(patient_id, doctor_id, transcript):
                    if not transcript.strip():
                        return {}, [], "Please transcribe audio or type a transcript first."
                    try:
                        structured = structure_transcript(transcript)
                        entities = extract_medical_entities(transcript)
                        record_id = save_record(
                            patient_id=patient_id or "UNKNOWN",
                            doctor_id=doctor_id or "DR001",
                            transcript=transcript,
                            structured=structured,
                            entities=entities,
                            language="mixed",
                        )
                        return structured, entities, f"✓ Saved. Record ID: {record_id}"
                    except Exception as e:
                        logging.exception("process error")
                        return {}, [], f"Error: {e}"

                # Only button click — no stop_recording
                transcribe_btn.click(
                    fn=on_transcribe,
                    inputs=[audio_input],
                    outputs=[transcript_box],
                )
                process_btn.click(
                    fn=on_process,
                    inputs=[patient_id_box, doctor_id_box, transcript_box],
                    outputs=[soap_box, entities_box, status_box],
                )

            # ── Tab 2: Doctor Dashboard ────────────────────────────
            with gr.Tab("📊  Dashboard"):

                gr.HTML('<p style="color:#64748B;font-size:0.84rem;padding:10px 0 6px;background:transparent;">Today\'s consultation summary for the active doctor.</p>')

                with gr.Group(elem_classes="vl-card"):
                    gr.HTML('<div class="vl-section-label">Overview</div>')
                    with gr.Row():
                        refresh_btn = gr.Button("↻  Refresh Dashboard", variant="secondary", scale=1)
                        with gr.Column(scale=2):
                            total_box = gr.Number(
                                label="Patients Seen Today",
                                elem_classes="vl-stat",
                            )

                with gr.Row(equal_height=False):
                    with gr.Column():
                        with gr.Group(elem_classes="vl-card"):
                            gr.HTML('<div class="vl-section-label">Top 5 Symptoms Today</div>')
                            symptoms_box = gr.JSON(label="", show_label=False)
                    with gr.Column():
                        with gr.Group(elem_classes="vl-card"):
                            gr.HTML('<div class="vl-section-label">Language Breakdown</div>')
                            lang_box = gr.JSON(label="", show_label=False)

                def refresh_dashboard(doctor_id):
                    try:
                        data = get_doctor_dashboard(doctor_id)
                        return (
                            data["total_patients_today"],
                            data["top_symptoms"],
                            data["languages"],
                        )
                    except Exception as e:
                        return 0, [], {"error": str(e)}

                # Use State, not the textbox from Tab 1
                refresh_btn.click(
                    fn=refresh_dashboard,
                    inputs=[doctor_id_state],
                    outputs=[total_box, symptoms_box, lang_box],
                )

            # ── Tab 3: Health Alerts ───────────────────────────────
            with gr.Tab("⚠  Health Alerts"):

                gr.HTML('<p style="color:#64748B;font-size:0.84rem;padding:10px 0 6px;background:transparent;">Population-level insights from Spark analytics. Run Notebook 03 to generate alerts.</p>')

                with gr.Group(elem_classes="vl-card"):
                    gr.HTML('<div class="vl-section-label">Active Alerts</div>')
                    alerts_refresh = gr.Button("↻  Load Alerts", variant="secondary")
                    alerts_display = gr.JSON(label="", show_label=False)

                def load_alerts():
                    query = """
                        SELECT alert_id, generated_at, insight_text, severity
                        FROM workspace.vaidya.health_alerts
                        ORDER BY generated_at DESC LIMIT 5
                    """
                    try:
                        with _get_sql_connection() as conn:
                            with conn.cursor() as cursor:
                                cursor.execute(query)
                                cols = [d[0] for d in cursor.description]
                                return [dict(zip(cols, row)) for row in cursor.fetchall()]
                    except Exception as e:
                        return [{"error": str(e)}]

                alerts_refresh.click(fn=load_alerts, outputs=[alerts_display])

    return demo

def main():
    logging.basicConfig(level="INFO")
    _load_secrets()
    demo = build_app()
    demo.queue()
    demo.launch()  # bare launch — platform injects all env vars

if __name__ == "__main__":
    main()





