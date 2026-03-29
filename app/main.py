from __future__ import annotations
import base64, json, logging, os, sys, uuid
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

CSS = """
        .section-header { font-size:15px; font-weight:600; color:#2d3748; margin:8px 0 4px }
        .stat-card       { background:#f0fff4; border-radius:8px; padding:12px 16px; text-align:center }
        .stat-num        { font-size:28px; font-weight:700; color:#276749 }
        .stat-label      { font-size:12px; color:#718096; margin-top:2px }
        footer           { visibility:hidden }

        /* ── Dark mode: dataframe tables ─────────────────────────────────── */
        .dark-table table,
        .dark .dark-table table {
            background: transparent !important;
        }
        .dark-table thead tr th,
        .dark .dark-table thead tr th {
            background: var(--background-fill-secondary) !important;
            color: var(--body-text-color) !important;
            border-bottom: 1px solid var(--border-color-primary) !important;
        }
        .dark-table tbody tr td,
        .dark .dark-table tbody tr td {
            background: var(--background-fill-primary) !important;
            color: var(--body-text-color) !important;
            border-bottom: 1px solid var(--border-color-primary) !important;
        }
        .dark-table tbody tr:hover td,
        .dark .dark-table tbody tr:hover td {
            background: var(--background-fill-secondary) !important;
        }

        /* ── Dark mode: textbox fields ───────────────────────────────────── */
        .dark-field textarea,
        .dark-field input,
        .dark .dark-field textarea,
        .dark .dark-field input {
            background: var(--background-fill-primary) !important;
            color: var(--body-text-color) !important;
            border: 1px solid var(--border-color-primary) !important;
        }
        .dark-field label span,
        .dark .dark-field label span {
            color: var(--body-text-color) !important;
        }

        /* ── Header logo + title ─────────────────────────────────────────── */
        .vaidya-header {
            display: flex;
            align-items: center;
            gap: 14px;
            padding: 8px 0 4px;
        }
        .vaidya-logo svg {
            width: 52px;
            height: 52px;
        }
        .vaidya-title {
            font-size: 26px;
            font-weight: 700;
            color: var(--body-text-color);
            line-height: 1.2;
        }
        .vaidya-subtitle {
            font-size: 13px;
            color: var(--body-text-color-subdued);
            margin-top: 2px;
        }

        /* ── Stat cards dark ─────────────────────────────────────────────── */
        @media (prefers-color-scheme: dark) {
            .stat-card  { background: var(--background-fill-secondary) }
            .stat-num   { color: #68d391 }
            .stat-label { color: var(--body-text-color-subdued) }
        }
        .dark .stat-card  { background: var(--background-fill-secondary) }
        .dark .stat-num   { color: #68d391 }
        .dark .stat-label { color: var(--body-text-color-subdued) }

        /* ── SOAP accent bars — green instead of blue ────────────────────── */
        .soap-s { border-left-color: #48bb78 !important; }
        .soap-o { border-left-color: #38a169 !important; }
        .soap-a { border-left-color: #276749 !important; }
        .soap-p { border-left-color: #9ae6b4 !important; }
        """


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



# ── HTML rendering helpers ────────────────────────────────────────────────────

def render_soap_html(structured: dict, entities: list) -> str:
    """Render structured note as a styled HTML card."""
    symptoms   = structured.get("symptoms", [])
    meds       = structured.get("medications", [])
    diagnosis  = structured.get("diagnosis", "—")
    plan       = structured.get("plan", "—")
    soap_s     = structured.get("soap_s", "—")
    soap_o     = structured.get("soap_o", "—")
    soap_a     = structured.get("soap_a", "—")
    soap_p     = structured.get("soap_p", "—")

    sym_html = "".join(f'<span style="background:#e8f4fd;border:1px solid #90cdf4;border-radius:12px;padding:3px 10px;margin:3px;display:inline-block;font-size:13px">{s}</span>' for s in symptoms) or "<em>None detected</em>"
    med_html = "".join(f'<span style="background:#f0fff4;border:1px solid #9ae6b4;border-radius:12px;padding:3px 10px;margin:3px;display:inline-block;font-size:13px">💊 {m}</span>' for m in meds) or "<em>None</em>"
    ent_html = "".join(f'<span style="background:#faf5ff;border:1px solid #d6bcfa;border-radius:12px;padding:3px 10px;margin:3px;display:inline-block;font-size:12px">🔬 {e["term"]} <code style="font-size:10px;color:#805ad5">{e["concept_id"]}</code></span>' for e in entities) or "<em>No entities mapped</em>"

    return f"""
    <div style="font-family:sans-serif;max-width:100%;padding:4px">

      <div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:16px;margin-bottom:12px">
        <div style="font-weight:600;color:#2d3748;margin-bottom:8px;font-size:15px">🏷️ Symptoms</div>
        <div>{sym_html}</div>
      </div>

      <div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:16px;margin-bottom:12px">
        <div style="font-weight:600;color:#2d3748;margin-bottom:8px;font-size:15px">💊 Medications</div>
        <div>{med_html}</div>
      </div>

      <div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:16px;margin-bottom:12px">
        <div style="font-weight:600;color:#2d3748;margin-bottom:4px;font-size:15px">🩺 Diagnosis</div>
        <div style="color:#4a5568;padding:4px 0">{diagnosis}</div>
      </div>

      <div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:16px;margin-bottom:12px">
        <div style="font-weight:600;color:#2d3748;margin-bottom:8px;font-size:15px">📋 SOAP Note</div>
        <table style="width:100%;border-collapse:collapse;font-size:13px">
          <tr><td style="padding:6px 10px;background:#f7fafc;border-radius:4px;width:28px;font-weight:600;color:#4299e1">S</td><td style="padding:6px 10px;color:#4a5568">{soap_s}</td></tr>
          <tr><td style="padding:6px 10px;background:#f7fafc;border-radius:4px;font-weight:600;color:#48bb78">O</td><td style="padding:6px 10px;color:#4a5568">{soap_o}</td></tr>
          <tr><td style="padding:6px 10px;background:#f7fafc;border-radius:4px;font-weight:600;color:#ed8936">A</td><td style="padding:6px 10px;color:#4a5568">{soap_a}</td></tr>
          <tr><td style="padding:6px 10px;background:#f7fafc;border-radius:4px;font-weight:600;color:#9f7aea">P</td><td style="padding:6px 10px;color:#4a5568">{soap_p}</td></tr>
        </table>
      </div>

      <div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:16px">
        <div style="font-weight:600;color:#2d3748;margin-bottom:8px;font-size:15px">🔬 SNOMED Entities</div>
        <div>{ent_html}</div>
      </div>
    </div>
    """

def fetch_patient_last_visit(patient_id: str) -> str:
    """Returns rendered HTML of last visit, or empty string if no history."""
    if not patient_id or len(patient_id.strip()) < 3:
        return ""
    try:
        with _get_sql_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT
                        timestamp,
                        get_json_object(structured_note, '$.diagnosis')   as diagnosis,
                        get_json_object(structured_note, '$.symptoms')     as symptoms,
                        get_json_object(structured_note, '$.medications')  as medications,
                        soap_plan,
                        doctor_id,
                        COUNT(*) OVER (PARTITION BY patient_id)            as total_visits
                    FROM workspace.vaidya.patient_records
                    WHERE patient_id = '{patient_id.strip()}'
                    ORDER BY timestamp DESC
                    LIMIT 1
                """)
                row = cur.fetchone()

        if not row:
            return "<div style='color:var(--body-text-color-subdued);font-size:13px;padding:8px'>No previous visits found for this patient.</div>"

        ts, diagnosis, symptoms_json, meds_json, plan, doctor, total = row
        ts_str = str(ts)[:16] if ts else "Unknown"

        # Parse JSON arrays safely
        try:
            symptoms = json.loads(symptoms_json) if symptoms_json else []
        except Exception:
            symptoms = []
        try:
            meds = json.loads(meds_json) if meds_json else []
        except Exception:
            meds = []

        sym_tags = "".join(
            f'<span style="background:var(--background-fill-secondary);border:1px solid var(--border-color-primary);border-radius:10px;padding:2px 9px;margin:2px;display:inline-block;font-size:12px;color:var(--body-text-color)">{s}</span>'
            for s in symptoms
        ) or "<em>—</em>"

        med_tags = "".join(
            f'<span style="background:var(--background-fill-secondary);border:1px solid var(--border-color-primary);border-radius:10px;padding:2px 9px;margin:2px;display:inline-block;font-size:12px;color:var(--body-text-color)">💊 {m}</span>'
            for m in meds
        ) or "<em>—</em>"

        return f"""
        <div style="border:1px solid var(--border-color-primary);border-radius:10px;
                    padding:14px 16px;background:var(--background-fill-secondary);margin-bottom:4px">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
                <span style="font-weight:600;font-size:14px;color:var(--body-text-color)">
                    📋 Last Visit — {ts_str}
                </span>
                <span style="font-size:12px;color:var(--body-text-color-subdued)">
                    {total} total visit{"s" if total != 1 else ""} · {doctor}
                </span>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
                <div>
                    <div style="font-size:11px;font-weight:600;color:var(--body-text-color-subdued);margin-bottom:4px">DIAGNOSIS</div>
                    <div style="font-size:13px;color:var(--body-text-color)">{diagnosis or "—"}</div>
                </div>
                <div>
                    <div style="font-size:11px;font-weight:600;color:var(--body-text-color-subdued);margin-bottom:4px">PLAN GIVEN</div>
                    <div style="font-size:13px;color:var(--body-text-color)">{(plan or "—")[:120]}{"..." if plan and len(plan) > 120 else ""}</div>
                </div>
                <div>
                    <div style="font-size:11px;font-weight:600;color:var(--body-text-color-subdued);margin-bottom:4px">SYMPTOMS</div>
                    <div>{sym_tags}</div>
                </div>
                <div>
                    <div style="font-size:11px;font-weight:600;color:var(--body-text-color-subdued);margin-bottom:4px">MEDICATIONS</div>
                    <div>{med_tags}</div>
                </div>
            </div>
        </div>
        """
    except Exception as e:
        return f"<div style='color:red;font-size:12px'>Could not load history: {e}</div>"


# ── Chart helpers ─────────────────────────────────────────────────────────────

def make_symptom_chart(top_symptoms: list, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not top_symptoms:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No data yet", ha="center", va="center", fontsize=13, color="#888")
        ax.axis("off")
        return fig

    labels = [s[0] for s in top_symptoms][::-1]
    values = [s[1] for s in top_symptoms][::-1]
    colors = ["#4299e1","#48bb78","#ed8936","#9f7aea","#fc8181"][:len(labels)]

    fig, ax = plt.subplots(figsize=(6, max(3, len(labels)*0.6)))
    bars = ax.barh(labels, values, color=colors, edgecolor="none", height=0.5)
    ax.bar_label(bars, padding=4, fontsize=11)
    ax.set_xlabel("Patients", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.spines[["top","right","left"]].set_visible(False)
    ax.tick_params(left=False)
    ax.set_xlim(0, max(values) * 1.25)
    fig.tight_layout()
    return fig


def make_language_chart(lang_dict: dict, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not lang_dict:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.text(0.5, 0.5, "No data yet", ha="center", va="center")
        ax.axis("off")
        return fig

    label_map = {"en-IN":"English","hi-IN":"Hindi","mr-IN":"Marathi",
                 "ta-IN":"Tamil","te-IN":"Telugu","kn-IN":"Kannada","mixed":"Mixed"}
    labels = [label_map.get(k, k) for k in lang_dict.keys()]
    values = list(lang_dict.values())
    colors = ["#4299e1","#48bb78","#ed8936","#9f7aea","#fc8181","#76e4f7","#f6ad55"]

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(values, labels=labels, autopct="%1.0f%%",
           colors=colors[:len(values)], startangle=90,
           wedgeprops={"edgecolor":"white","linewidth":2})
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def make_daily_volume_chart(rows: list, title: str):
    """rows: list of (date_str, count)"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import OrderedDict

    if not rows:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No data yet", ha="center", va="center")
        ax.axis("off")
        return fig

    data = OrderedDict(sorted(rows))
    dates = list(data.keys())
    counts = list(data.values())

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(dates, counts, marker="o", color="#4299e1", linewidth=2, markersize=6)
    ax.fill_between(dates, counts, alpha=0.15, color="#4299e1")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("Patients")
    ax.spines[["top","right"]].set_visible(False)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    fig.tight_layout()
    return fig


# ── Dashboard data fetchers ───────────────────────────────────────────────────

def fetch_all_records_for_ml() -> list[dict]:
    """Pull all records needed for clustering."""
    with _get_sql_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    record_id,
                    patient_id,
                    doctor_id,
                    DATE(timestamp)                                      AS visit_date,
                    timestamp,
                    get_json_object(structured_note, '$.diagnosis')      AS diagnosis,
                    get_json_object(structured_note, '$.symptoms')       AS symptoms_json,
                    get_json_object(structured_note, '$.medications')    AS medications_json
                FROM workspace.vaidya.patient_records
                WHERE structured_note IS NOT NULL
                ORDER BY timestamp DESC
            """)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]


def run_ml_analysis(records: list[dict], n_clusters: int = 4):
    """
    Full ML pipeline:
      1. Build binary symptom vectors
      2. K-Means clustering
      3. Anomaly detection via centroid distance
      4. Temporal spike detection
    Returns a dict of matplotlib figures.
    """
    import json
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import Counter
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA

    GREEN = ["#1a4731","#276749","#2f855a","#38a169","#48bb78","#68d391","#9ae6b4","#c6f6d5"]

    # ── 1. Parse symptoms ─────────────────────────────────────────────────────
    parsed = []
    for r in records:
        try:
            syms = json.loads(r.get("symptoms_json") or "[]")
            syms = [s.lower().strip() for s in syms if s]
        except Exception:
            syms = []
        if syms:
            parsed.append({**r, "symptoms": syms})

    if len(parsed) < 4:
        return None, "Not enough records with symptoms (need at least 4)."

    symptom_lists = [r["symptoms"] for r in parsed]

    # ── 2. Binary vectorisation ───────────────────────────────────────────────
    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(symptom_lists)
    vocab = mlb.classes_
    n_clusters = min(n_clusters, len(parsed) - 1)

    # ── 3. Silhouette sweep ───────────────────────────────────────────────────
    k_range = range(2, min(8, len(parsed)))
    sil_scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=200)
        labels = km.fit_predict(X)
        if len(set(labels)) > 1:
            sil_scores[k] = silhouette_score(X, labels, metric="cosine")

    best_k = max(sil_scores, key=sil_scores.get) if sil_scores else n_clusters

    # ── 4. Final K-Means ─────────────────────────────────────────────────────
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10, max_iter=300)
    cluster_labels = km_final.fit_predict(X)
    centroids = km_final.cluster_centers_

    for i, r in enumerate(parsed):
        r["cluster"] = int(cluster_labels[i])

    # ── 5. Anomaly score = cosine distance to assigned centroid ───────────────
    def cosine_dist(vec, centroid):
        dot  = np.dot(vec, centroid)
        norm = np.linalg.norm(vec) * np.linalg.norm(centroid)
        return 1.0 - dot / norm if norm > 0 else 1.0

    scores = np.array([cosine_dist(X[i], centroids[cluster_labels[i]])
                       for i in range(len(parsed))])
    mean_s, std_s = scores.mean(), scores.std()
    threshold = mean_s + 1.5 * std_s
    anomaly_flags = scores > threshold

    for i, r in enumerate(parsed):
        r["anomaly_score"] = float(scores[i])
        r["is_anomaly"]    = bool(anomaly_flags[i])

    # ── 6. Temporal daily counts ──────────────────────────────────────────────
    from collections import defaultdict
    daily_sym: dict = defaultdict(lambda: defaultdict(int))
    for r in parsed:
        day = str(r.get("visit_date", ""))[:10]
        for s in r["symptoms"]:
            daily_sym[day][s] += 1

    # ══ FIGURES ══════════════════════════════════════════════════════════════

    # Fig 1 — Silhouette scores
    fig1, ax = plt.subplots(figsize=(7, 3.5))
    ks = list(sil_scores.keys())
    ss = list(sil_scores.values())
    bar_colors = [GREEN[3] if k == best_k else GREEN[5] for k in ks]
    bars = ax.bar(ks, ss, color=bar_colors, edgecolor="none", width=0.5)
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
    ax.axvline(best_k, color=GREEN[1], linestyle="--", linewidth=1.5,
               label=f"Best K={best_k} ({sil_scores[best_k]:.3f})")
    ax.set_xlabel("Number of Clusters (K)", fontsize=11)
    ax.set_ylabel("Silhouette Score", fontsize=11)
    ax.set_title("K-Means: Silhouette Score by K", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
    ax.set_ylim(0, max(ss) * 1.25 if ss else 1)
    fig1.tight_layout()

    # Fig 2 — Cluster symptom profiles (horizontal bars)
    cluster_profiles = {}
    for cid in range(best_k):
        members = [r for r in parsed if r["cluster"] == cid]
        all_sym = [s for r in members for s in r["symptoms"]]
        top = Counter(all_sym).most_common(6)
        top_dx = Counter([r.get("diagnosis","?") for r in members
                          if r.get("diagnosis")]).most_common(2)
        cluster_profiles[cid] = {
            "size": len(members),
            "top_symptoms": top,
            "top_diagnoses": [d for d,_ in top_dx],
        }

    fig2, axes = plt.subplots(1, best_k,
                              figsize=(max(5 * best_k, 10), 5),
                              sharey=False)
    if best_k == 1:
        axes = [axes]
    for ax, (cid, prof) in zip(axes, cluster_profiles.items()):
        syms   = [s for s,_ in prof["top_symptoms"]][::-1]
        counts = [c for _,c in prof["top_symptoms"]][::-1]
        color  = GREEN[2 + cid % 4]
        b = ax.barh(syms, counts, color=color, edgecolor="none", height=0.55)
        ax.bar_label(b, padding=3, fontsize=9)
        dx_label = ", ".join(prof["top_diagnoses"][:2]) or "—"
        ax.set_title(
            f"Cluster {cid}  ({prof['size']} patients)\n"
            f"📋 {dx_label}",
            fontsize=10, fontweight="bold", color=GREEN[1]
        )
        ax.set_xlabel("Frequency", fontsize=9)
        ax.spines[["top","right","left"]].set_visible(False)
        ax.tick_params(left=False)
        ax.set_xlim(0, max(counts) * 1.3 if counts else 1)
    fig2.suptitle("Symptom Clusters — What groups of patients look alike?",
                  fontsize=13, fontweight="bold")
    fig2.tight_layout()

    # Fig 3 — PCA scatter (2D projection of clusters)
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)
    fig3, ax = plt.subplots(figsize=(7, 5))
    for cid in range(best_k):
        mask = cluster_labels == cid
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=GREEN[2 + cid % 4], label=f"Cluster {cid}",
                   alpha=0.75, edgecolors="white", linewidths=0.5, s=60)
    # Mark anomalies with red rings
    anom_mask = np.array([r["is_anomaly"] for r in parsed])
    ax.scatter(X_2d[anom_mask, 0], X_2d[anom_mask, 1],
               facecolors="none", edgecolors="#e53e3e",
               linewidths=1.8, s=130, label="⚠ Anomaly", zorder=5)
    ax.set_title("Patient Clusters — PCA Projection\n(red rings = anomalies)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    ax.legend(fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
    fig3.tight_layout()

    # Fig 4 — Anomaly score distribution
    fig4, ax = plt.subplots(figsize=(7, 4))
    for cid in range(best_k):
        mask = cluster_labels == cid
        ax.hist(scores[mask], bins=12, alpha=0.6,
                color=GREEN[2 + cid % 4], label=f"Cluster {cid}", edgecolor="none")
    ax.axvline(threshold, color="#e53e3e", linewidth=2, linestyle="--",
               label=f"Anomaly threshold ({threshold:.3f})")
    ax.axvline(mean_s, color=GREEN[1], linewidth=1.5, linestyle=":",
               label=f"Mean ({mean_s:.3f})")
    ax.set_xlabel("Anomaly Score (cosine distance to centroid)", fontsize=11)
    ax.set_ylabel("Number of records", fontsize=11)
    ax.set_title(f"Anomaly Distribution — {anomaly_flags.sum()} anomalies flagged",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
    fig4.tight_layout()

    # Fig 5 — Temporal symptom spikes (top 5 symptoms over time)
    all_sym_flat = Counter([s for r in parsed for s in r["symptoms"]])
    top5 = [s for s, _ in all_sym_flat.most_common(5)]
    days_sorted = sorted(daily_sym.keys())

    fig5, ax = plt.subplots(figsize=(9, 4))
    for i, sym in enumerate(top5):
        counts_by_day = [daily_sym[d].get(sym, 0) for d in days_sorted]
        if max(counts_by_day) == 0:
            continue
        ax.plot(days_sorted, counts_by_day,
                marker="o", markersize=5, linewidth=2,
                color=GREEN[1 + i], label=sym)

        # Spike detection: flag days > mean + 2σ for this symptom
        arr = np.array(counts_by_day, dtype=float)
        if arr.std() > 0:
            spike_thresh = arr.mean() + 2 * arr.std()
            for j, (day, val) in enumerate(zip(days_sorted, counts_by_day)):
                if val > spike_thresh:
                    ax.annotate("⚠", xy=(day, val),
                                fontsize=13, color="#e53e3e",
                                ha="center", va="bottom")

    ax.set_title("Symptom Trends Over Time\n(⚠ = statistical spike, >2σ above mean)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Cases per day", fontsize=10)
    ax.legend(fontsize=9, loc="upper left")
    ax.spines[["top","right"]].set_visible(False)
    plt.xticks(rotation=35, ha="right", fontsize=8)
    fig5.tight_layout()

    # Anomaly table data
    anomaly_rows = [
        [r["patient_id"], r["doctor_id"],
         ", ".join(r["symptoms"][:3]),
         r.get("diagnosis","—"),
         f"{r['anomaly_score']:.3f}",
         f"Cluster {r['cluster']}"]
        for r in sorted(parsed, key=lambda x: -x["anomaly_score"])
        if r["is_anomaly"]
    ]

    summary = (
        f"**{len(parsed)}** records analysed · "
        f"**{best_k}** clusters found · "
        f"**{anomaly_flags.sum()}** anomalies flagged "
        f"(threshold {threshold:.3f})"
    )

    return {
        "fig_silhouette":   fig1,
        "fig_clusters":     fig2,
        "fig_pca":          fig3,
        "fig_anomaly_dist": fig4,
        "fig_temporal":     fig5,
        "anomaly_rows":     anomaly_rows,
        "summary":          summary,
    }, None


def fetch_dashboard_data(doctor_id: str, scope: str = "personal") -> dict:
    """scope = 'personal' (one doctor) or 'regional' (all doctors)."""
    where = f"doctor_id = '{doctor_id}'" if scope == "personal" else "1=1"

    queries = {
        "total_today": f"""
            SELECT COUNT(*) FROM workspace.vaidya.patient_records
            WHERE {where} AND DATE(timestamp) = CURRENT_DATE()""",

        "total_week": f"""
            SELECT COUNT(*) FROM workspace.vaidya.patient_records
            WHERE {where} AND timestamp >= current_timestamp() - interval 7 days""",

        "top_symptoms": f"""
            SELECT symptom, COUNT(*) as cnt FROM (
                SELECT explode(from_json(structured_note,
                    'struct<symptoms:array<string>>').symptoms) as symptom
                FROM workspace.vaidya.patient_records
                WHERE {where} AND timestamp >= current_timestamp() - interval 7 days
            ) WHERE symptom IS NOT NULL
            GROUP BY symptom ORDER BY cnt DESC LIMIT 8""",

        "languages": f"""
            SELECT language_detected, COUNT(*) as cnt
            FROM workspace.vaidya.patient_records
            WHERE {where} AND timestamp >= current_timestamp() - interval 7 days
            GROUP BY language_detected""",

        "daily_volume": f"""
            SELECT DATE(timestamp) as day, COUNT(*) as cnt
            FROM workspace.vaidya.patient_records
            WHERE {where} AND timestamp >= current_timestamp() - interval 7 days
            GROUP BY DATE(timestamp) ORDER BY day""",

        "top_diagnoses": f"""
            SELECT
                get_json_object(structured_note,'$.diagnosis') as diagnosis,
                COUNT(*) as cnt
            FROM workspace.vaidya.patient_records
            WHERE {where} AND timestamp >= current_timestamp() - interval 7 days
            GROUP BY get_json_object(structured_note,'$.diagnosis')
            ORDER BY cnt DESC LIMIT 5""",
    }

    if scope == "regional":
        queries["doctor_volume"] = f"""
            SELECT doctor_id, COUNT(*) as cnt
            FROM workspace.vaidya.patient_records
            WHERE timestamp >= current_timestamp() - interval 7 days
            GROUP BY doctor_id ORDER BY cnt DESC"""

    results = {}
    with _get_sql_connection() as conn:
        with conn.cursor() as cur:
            for key, q in queries.items():
                cur.execute(q)
                results[key] = cur.fetchall()

    return results


def fetch_candidate_alerts() -> list[dict]:
    with _get_sql_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT alert_id, insight_text, severity, generated_at
                FROM workspace.vaidya.health_alerts
                ORDER BY generated_at DESC LIMIT 10
            """)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]


def publish_alerts(selected_texts: list[str], doctor_id: str):
    if not selected_texts:
        return
    with _get_sql_connection() as conn:
        with conn.cursor() as cur:
            for text in selected_texts:
                alert_id = str(uuid.uuid4())
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                def esc(s): return s.replace("'","''")
                cur.execute(f"""
                    INSERT INTO workspace.vaidya.health_alerts_published
                    (alert_id, published_at, insight_text, severity, published_by)
                    VALUES ('{alert_id}','{now}','{esc(text)}','INFO','{doctor_id}')
                """)


def fetch_published_alerts() -> list[dict]:
    with _get_sql_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT insight_text, severity, published_at, published_by
                FROM workspace.vaidya.health_alerts_published
                ORDER BY published_at DESC LIMIT 10
            """)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]


def fetch_records_filtered(date_str: str, patient_id: str, doctor_id: str, scope: str) -> list:
    """Fetch records filtered by date, patient, doctor scope."""
    conditions = []

    if date_str:
        conditions.append(f"DATE(timestamp) = '{date_str}'")

    if patient_id and patient_id.strip():
        conditions.append(f"patient_id = '{patient_id.strip()}'")

    if scope == "mine":
        conditions.append(f"doctor_id = '{doctor_id}'")

    where = "WHERE " + " AND ".join(conditions) if conditions else ""

    query = f"""
        SELECT
            patient_id,
            doctor_id,
            DATE(timestamp)                                         AS visit_date,
            CAST(timestamp AS STRING)                               AS visit_time,
            get_json_object(structured_note, '$.diagnosis')        AS diagnosis,
            get_json_object(structured_note, '$.symptoms')         AS symptoms_json,
            get_json_object(structured_note, '$.medications')      AS medications_json,
            soap_subjective,
            soap_objective,
            soap_assessment,
            soap_plan,
            language_detected,
            record_id
        FROM workspace.vaidya.patient_records
        {where}
        ORDER BY timestamp DESC
        LIMIT 200
    """
    try:
        with _get_sql_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]
    except Exception as e:
        return [{"error": str(e)}]


def render_records_html(records: list) -> str:
    if not records:
        return "<div style='padding:20px;color:var(--body-text-color-subdued);text-align:center'>No records found for the selected filters.</div>"

    if records and "error" in records[0]:
        return f"<div style='color:red;padding:12px'>Error: {records[0]['error']}</div>"

    cards = ""
    for r in records:
        # Parse symptoms and medications
        try:
            symptoms = json.loads(r.get("symptoms_json") or "[]")
        except Exception:
            symptoms = []
        try:
            meds = json.loads(r.get("medications_json") or "[]")
        except Exception:
            meds = []

        sym_tags = "".join(
            f'<span style="background:var(--background-fill-secondary);border:1px solid var(--border-color-primary);'
            f'border-radius:10px;padding:2px 9px;margin:2px;display:inline-block;font-size:12px;'
            f'color:var(--body-text-color)">{s}</span>'
            for s in symptoms
        ) or "<em style='color:var(--body-text-color-subdued)'>None</em>"

        med_tags = "".join(
            f'<span style="background:var(--background-fill-secondary);border:1px solid var(--border-color-primary);'
            f'border-radius:10px;padding:2px 9px;margin:2px;display:inline-block;font-size:12px;'
            f'color:var(--body-text-color)">💊 {m}</span>'
            for m in meds
        ) or "<em style='color:var(--body-text-color-subdued)'>None</em>"

        lang_map = {"en-IN":"🇬🇧 EN","hi-IN":"🇮🇳 HI","mr-IN":"🇮🇳 MR",
                    "ta-IN":"🇮🇳 TA","te-IN":"🇮🇳 TE","kn-IN":"🇮🇳 KN","mixed":"🔀 Mix"}
        lang_badge = lang_map.get(r.get("language_detected",""), r.get("language_detected",""))

        ts = str(r.get("visit_time",""))[:16]
        diagnosis = r.get("diagnosis") or "—"
        soap_s = r.get("soap_subjective") or "—"
        soap_o = r.get("soap_objective")  or "—"
        soap_a = r.get("soap_assessment") or "—"
        soap_p = r.get("soap_plan")       or "—"
        record_id = str(r.get("record_id",""))[:8]

        cards += f"""
        <div style="border:1px solid var(--border-color-primary);border-radius:12px;
                    padding:16px;margin-bottom:14px;background:var(--background-fill-secondary)">

          <!-- Header row -->
          <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px">
            <div>
              <span style="font-size:16px;font-weight:600;color:var(--body-text-color)">
                👤 {r.get("patient_id","—")}
              </span>
              <span style="margin-left:10px;font-size:12px;color:var(--body-text-color-subdued)">
                Dr. {r.get("doctor_id","—")}
              </span>
            </div>
            <div style="text-align:right">
              <span style="font-size:12px;color:var(--body-text-color-subdued)">{ts}</span>
              <span style="margin-left:8px;font-size:11px;background:var(--background-fill-primary);
                           border:1px solid var(--border-color-primary);border-radius:6px;
                           padding:2px 6px;color:var(--body-text-color-subdued)">{lang_badge}</span>
              <span style="margin-left:6px;font-size:10px;color:var(--body-text-color-subdued)">#{record_id}</span>
            </div>
          </div>

          <!-- Diagnosis banner -->
          <div style="background:var(--background-fill-primary);border-left:3px solid #4299e1;
                      border-radius:0 6px 6px 0;padding:8px 12px;margin-bottom:12px">
            <span style="font-size:11px;font-weight:600;color:#4299e1;letter-spacing:0.5px">DIAGNOSIS</span>
            <div style="font-size:14px;color:var(--body-text-color);margin-top:2px">{diagnosis}</div>
          </div>

          <!-- Symptoms + Medications side by side -->
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px">
            <div>
              <div style="font-size:11px;font-weight:600;color:var(--body-text-color-subdued);
                          letter-spacing:0.5px;margin-bottom:6px">🏷️ SYMPTOMS</div>
              <div>{sym_tags}</div>
            </div>
            <div>
              <div style="font-size:11px;font-weight:600;color:var(--body-text-color-subdued);
                          letter-spacing:0.5px;margin-bottom:6px">💊 MEDICATIONS</div>
              <div>{med_tags}</div>
            </div>
          </div>

          <!-- SOAP grid -->
          <div style="font-size:11px;font-weight:600;color:var(--body-text-color-subdued);
                      letter-spacing:0.5px;margin-bottom:6px">📋 SOAP NOTE</div>
          <table style="width:100%;border-collapse:collapse;font-size:13px">
            <tr>
              <td style="width:22px;padding:5px 10px 5px 8px;font-weight:700;color:#4299e1;
                         vertical-align:top;border-bottom:1px solid var(--border-color-primary)">S</td>
              <td style="padding:5px 8px;color:var(--body-text-color);
                         border-bottom:1px solid var(--border-color-primary)">{soap_s}</td>
            </tr>
            <tr>
              <td style="padding:5px 10px 5px 8px;font-weight:700;color:#48bb78;
                         vertical-align:top;border-bottom:1px solid var(--border-color-primary)">O</td>
              <td style="padding:5px 8px;color:var(--body-text-color);
                         border-bottom:1px solid var(--border-color-primary)">{soap_o}</td>
            </tr>
            <tr>
              <td style="padding:5px 10px 5px 8px;font-weight:700;color:#ed8936;
                         vertical-align:top;border-bottom:1px solid var(--border-color-primary)">A</td>
              <td style="padding:5px 8px;color:var(--body-text-color);
                         border-bottom:1px solid var(--border-color-primary)">{soap_a}</td>
            </tr>
            <tr>
              <td style="padding:5px 10px 5px 8px;font-weight:700;color:#9f7aea;
                         vertical-align:top">P</td>
              <td style="padding:5px 8px;color:var(--body-text-color)">{soap_p}</td>
            </tr>
          </table>

        </div>
        """

    count_line = f'<div style="font-size:12px;color:var(--body-text-color-subdued);margin-bottom:12px">{len(records)} record(s) shown</div>'
    return count_line + cards

# ── Main app ──────────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:

    CSS = """
    .section-header {font-size:15px;font-weight:600;color:#2d3748;margin:8px 0 4px}
    .stat-card {background:#f7fafc;border-radius:8px;padding:12px 16px;text-align:center}
    .stat-num {font-size:28px;font-weight:700;color:#2b6cb0}
    .stat-label {font-size:12px;color:#718096;margin-top:2px}
    footer {visibility:hidden}
    """

    with gr.Blocks(
        theme=gr.themes.Soft(primary_hue="green", secondary_hue="emerald"),
        css=CSS,
        title="Vaidya Lipi — Medical Scribe"
    ) as demo:

        DOCTOR_ID = "DR001"   # in production this comes from login

        # gr.Markdown(
        #     "# Vaidya Lipi · वैद्य लिपि\n"
        #     "*AI Medical Scribe · ABDM-compatible · "
        #     f"Logged in as **{DOCTOR_ID}** (Head of Department)*"
        # )

        # REMOVE this line:
        gr.Markdown("# Vaidya Lipi · वैद्य लिपि\n*AI Medical Scribe ...*")

        # # REPLACE with:
        # gr.HTML(f"""
        # <div class="vaidya-header">
        #   <div class="vaidya-logo">
        #     # Change viewBox and add width/height directly on the svg tag:
        #     <svg viewBox="0 0 52 52" width="36" height="36" fill="none" xmlns="http://www.w3.org/2000/svg">
        #       <!-- Outer circle -->
        #       <circle cx="26" cy="26" r="25" fill="#f0fff4" stroke="#38a169" stroke-width="2"/>
        #       <!-- Doctor head -->
        #       <circle cx="26" cy="17" r="7" fill="#38a169"/>
        #       <!-- Doctor body / coat -->
        #       <path d="M13 42 C13 32 20 28 26 28 C32 28 39 32 39 42Z" fill="#38a169"/>
        #       <!-- Stethoscope -->
        #       <path d="M20 30 Q18 36 20 40 Q22 44 26 44 Q30 44 32 40 Q34 36 32 30"
        #             stroke="#276749" stroke-width="2" fill="none" stroke-linecap="round"/>
        #       <circle cx="26" cy="44" r="2.5" fill="#276749"/>
        #       <!-- Cross on coat -->
        #       <rect x="24.5" y="31" width="3" height="8" rx="1" fill="white"/>
        #       <rect x="22" y="33.5" width="8" height="3" rx="1" fill="white"/>
        #     </svg>
        #   </div>
        #   <div>
        #     <div class="vaidya-title">Vaidya Lipi · वैद्य लिपि</div>
        #     <div class="vaidya-subtitle">
        #       AI Medical Scribe · ABDM-compatible ·
        #       Logged in as <strong>{DOCTOR_ID}</strong> (Head of Department)
        #     </div>
        #   </div>
        # </div>
        # """)

        doctor_id_state = gr.State(DOCTOR_ID)
        # Store structured result between Process and Save steps
        structured_state = gr.State({})
        entities_state   = gr.State([])

        with gr.Tab("📝 Record Consultation"):

            with gr.Row():
                patient_id_box = gr.Textbox(
                    label="Patient ID (ABHA ID or local ID)",
                    placeholder="e.g. PAT1234 or 14-digit ABHA",
                    scale=3
                )
                doctor_id_box = gr.Textbox(
                    label="Doctor ID",
                    value=DOCTOR_ID,
                    scale=1
                )

            # ── Patient history panel — auto-populates on ID entry ─────────────
            last_visit_panel = gr.HTML(
                value="<div style='color:var(--body-text-color-subdued);font-size:13px;padding:6px 2px'>Enter a Patient ID above to see visit history.</div>"
            )

            # Wire: as soon as doctor finishes typing the patient ID, fetch history
            patient_id_box.change(
                fn=fetch_patient_last_visit,
                inputs=[patient_id_box],
                outputs=[last_visit_panel],
                show_progress="hidden",   # silent — don't flash a bar for a quick lookup
            )

            doctor_id_box.change(lambda x: x, inputs=[doctor_id_box], outputs=[doctor_id_state])


            audio_input = gr.Audio(
                sources=["microphone"],
                type="numpy",
                label="🎙️ Record consultation (Hindi, English, or any Indian language)",
            )

            transcript_box = gr.Textbox(
                label="Transcript — auto-filled after recording, edit freely before processing",
                lines=4,
                placeholder="Transcript appears here. You can also type directly."
            )

            with gr.Row():
                transcribe_btn = gr.Button("🎙️ Transcribe Audio",     variant="secondary", scale=1)
                process_btn    = gr.Button("⚙️  Analyse Transcript",   variant="secondary", scale=1)
                save_btn       = gr.Button("💾  Save Record",          variant="primary",   scale=1, interactive=False)

            # ── Results — always rendered, start empty ─────────────────────────
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🏷️ Symptoms")
                    symptoms_box = gr.Dataframe(
                        headers=["Symptom"],
                        datatype=["str"],
                        row_count=(1, "dynamic"),
                        interactive=False,
                        label=None,
                        elem_classes=["dark-table"],
                    )
                    gr.Markdown("### 💊 Medications")
                    medications_box = gr.Dataframe(
                        headers=["Medication"],
                        datatype=["str"],
                        row_count=(1, "dynamic"),
                        interactive=False,
                        label=None,
                        elem_classes=["dark-table"],
                    )
                    gr.Markdown("### 🩺 Diagnosis")
                    diagnosis_box = gr.Textbox(
                        label=None,
                        interactive=False,
                        lines=2,
                        elem_classes=["dark-field"],
                    )

                with gr.Column():
                    gr.Markdown("### 📋 SOAP Note")
                    soap_s_box = gr.Textbox(label="S — Subjective", interactive=False, lines=2, elem_classes=["dark-field"])
                    soap_o_box = gr.Textbox(label="O — Objective",  interactive=False, lines=2, elem_classes=["dark-field"])
                    soap_a_box = gr.Textbox(label="A — Assessment", interactive=False, lines=2, elem_classes=["dark-field"])
                    soap_p_box = gr.Textbox(label="P — Plan",       interactive=False, lines=3, elem_classes=["dark-field"])

                    gr.Markdown("### 🔬 SNOMED Entities")
                    entities_box_display = gr.Dataframe(
                        headers=["Term", "SNOMED Code", "Score"],
                        datatype=["str", "str", "number"],
                        row_count=(1, "dynamic"),
                        interactive=False,
                        label=None,
                        elem_classes=["dark-table"],
                    )

            save_status = gr.Textbox(label="Status", interactive=False, elem_classes=["dark-field"])

            # ── Handlers ───────────────────────────────────────────────────────
            def on_transcribe(audio):
                if audio is None:
                    return "No audio recorded. Please use the microphone."
                try:
                    transcript, lang = transcribe_audio(audio)
                    return f"[{lang}] {transcript}"
                except Exception as e:
                    return f"Transcription error: {e}"

            def on_process(transcript, progress=gr.Progress()):
                empty = ([], [], "", "", "", "", "", [], {}, [], gr.update(interactive=False), "")
                if not transcript.strip():
                    return empty
                try:
                    progress(0,   desc="Sending transcript to LLM...")
                    structured = structure_transcript(transcript)

                    progress(0.6, desc="Extracting SNOMED entities with Parrotlet-e...")
                    entities = extract_medical_entities(transcript)

                    progress(0.9, desc="Rendering results...")

                    symptoms  = [[s] for s in structured.get("symptoms", [])]
                    meds      = [[m] for m in structured.get("medications", [])]
                    ent_rows  = [[e["term"], e["concept_id"], round(e["score"], 3)] for e in entities]

                    progress(1.0, desc="Done")
                    return (
                        symptoms,
                        meds,
                        structured.get("diagnosis", ""),
                        structured.get("soap_s", ""),
                        structured.get("soap_o", ""),
                        structured.get("soap_a", ""),
                        structured.get("soap_p", ""),
                        ent_rows,
                        structured,
                        entities,
                        gr.update(interactive=True),
                        "",
                    )
                except Exception as e:
                    logging.exception("process error")
                    return ([], [], f"Error: {e}", "", "", "", "", [], {}, [], gr.update(interactive=False), f"❌ {e}")

            def on_save(patient_id, doctor_id, transcript, structured, entities):
                if not structured:
                    return "Nothing to save — analyse first."
                try:
                    record_id = save_record(
                        patient_id=patient_id or "UNKNOWN",
                        doctor_id=doctor_id or DOCTOR_ID,
                        transcript=transcript,
                        structured=structured,
                        entities=entities,
                        language="mixed"
                    )
                    return f"✅ Saved. Record ID: {record_id}"
                except Exception as e:
                    logging.exception("save error")
                    return f"❌ Save failed: {e}"

            transcribe_btn.click(
                fn=on_transcribe,
                inputs=[audio_input],
                outputs=[transcript_box],
                show_progress="full",
            )
            process_btn.click(
                fn=on_process,
                inputs=[transcript_box],
                outputs=[
                    symptoms_box, medications_box, diagnosis_box,
                    soap_s_box, soap_o_box, soap_a_box, soap_p_box,
                    entities_box_display,
                    structured_state, entities_state,
                    save_btn, save_status,
                ],
                show_progress="full",
            )
            save_btn.click(
                fn=on_save,
                inputs=[patient_id_box, doctor_id_box, transcript_box,
                        structured_state, entities_state],
                outputs=[save_status],
                show_progress="full",
            )

        # ── Tab 2: My Dashboard ───────────────────────────────────────────────
        with gr.Tab("📊 My Dashboard"):

            with gr.Row():
                my_today  = gr.HTML('<div class="stat-card"><div class="stat-num">—</div><div class="stat-label">Patients Today</div></div>')
                my_week   = gr.HTML('<div class="stat-card"><div class="stat-num">—</div><div class="stat-label">Patients This Week</div></div>')

            my_refresh = gr.Button("🔄 Refresh", variant="secondary")

            with gr.Row():
                my_symptom_chart = gr.Plot(label="Top Symptoms (7 days)")
                my_lang_chart    = gr.Plot(label="Language Mix (7 days)")

            my_volume_chart   = gr.Plot(label="Daily Patient Volume (7 days)")

            with gr.Row():
                my_diag_chart = gr.Plot(label="Top Diagnoses (7 days)")

            def refresh_my(doctor_id):
                try:
                    d = fetch_dashboard_data(doctor_id, scope="personal")
                    total_today = d["total_today"][0][0] if d["total_today"] else 0
                    total_week  = d["total_week"][0][0]  if d["total_week"]  else 0

                    today_html = f'<div class="stat-card"><div class="stat-num">{total_today}</div><div class="stat-label">Patients Today</div></div>'
                    week_html  = f'<div class="stat-card"><div class="stat-num">{total_week}</div><div class="stat-label">Patients This Week</div></div>'

                    sym_fig  = make_symptom_chart(d["top_symptoms"], "My Top Symptoms")
                    lang_fig = make_language_chart({r[0]:r[1] for r in d["languages"]}, "Language Mix")
                    vol_fig  = make_daily_volume_chart([(str(r[0]),r[1]) for r in d["daily_volume"]], "Daily Volume")
                    diag_fig = make_symptom_chart(d["top_diagnoses"], "My Top Diagnoses")

                    return today_html, week_html, sym_fig, lang_fig, vol_fig, diag_fig
                except Exception as e:
                    err = f'<div style="color:red">Error: {e}</div>'
                    empty = make_symptom_chart([], "")
                    return err, err, empty, empty, empty, empty

            my_refresh.click(
                fn=refresh_my,
                inputs=[doctor_id_state],
                outputs=[my_today, my_week, my_symptom_chart, my_lang_chart,
                         my_volume_chart, my_diag_chart]
            )

        # ── Tab 3: Regional Dashboard ─────────────────────────────────────────
        with gr.Tab("🏥 Regional Dashboard"):

            with gr.Row():
                reg_today = gr.HTML('<div class="stat-card"><div class="stat-num">—</div><div class="stat-label">Total Patients Today</div></div>')
                reg_week  = gr.HTML('<div class="stat-card"><div class="stat-num">—</div><div class="stat-label">Total Patients This Week</div></div>')

            reg_refresh = gr.Button("🔄 Refresh", variant="secondary")

            with gr.Row():
                reg_symptom_chart = gr.Plot(label="Hospital-wide Top Symptoms")
                reg_lang_chart    = gr.Plot(label="Language Mix Across Hospital")

            with gr.Row():
                reg_volume_chart = gr.Plot(label="Hospital Daily Volume")
                reg_doctor_chart = gr.Plot(label="Patients per Doctor")

            gr.Markdown("---\n### 🚨 Generate & Send Health Alerts")
            gr.Markdown("*Select candidate alerts from the AI analysis below to broadcast to all doctors.*")

            gen_alerts_btn  = gr.Button("⚙️ Load Candidate Alerts", variant="secondary")
            alert_choices   = gr.CheckboxGroup(choices=[], label="Select alerts to publish")
            publish_btn     = gr.Button("📢 Publish Selected Alerts", variant="primary")
            publish_status  = gr.Textbox(label="", interactive=False)

            def refresh_regional(doctor_id):
                try:
                    d = fetch_dashboard_data(doctor_id, scope="regional")
                    total_today = d["total_today"][0][0] if d["total_today"] else 0
                    total_week  = d["total_week"][0][0]  if d["total_week"]  else 0

                    today_html = f'<div class="stat-card"><div class="stat-num">{total_today}</div><div class="stat-label">Total Patients Today</div></div>'
                    week_html  = f'<div class="stat-card"><div class="stat-num">{total_week}</div><div class="stat-label">Total Patients This Week</div></div>'

                    sym_fig  = make_symptom_chart(d["top_symptoms"], "Hospital-wide Top Symptoms")
                    lang_fig = make_language_chart({r[0]:r[1] for r in d["languages"]}, "Language Mix")
                    vol_fig  = make_daily_volume_chart([(str(r[0]),r[1]) for r in d["daily_volume"]], "Daily Volume")
                    doc_fig  = make_symptom_chart(d.get("doctor_volume",[]), "Patients per Doctor")

                    return today_html, week_html, sym_fig, lang_fig, vol_fig, doc_fig
                except Exception as e:
                    err = f'<div style="color:red">Error: {e}</div>'
                    empty = make_symptom_chart([], "")
                    return err, err, empty, empty, empty, empty

            def load_candidate_alerts():
                try:
                    alerts = fetch_candidate_alerts()
                    choices = [a["insight_text"] for a in alerts]
                    return gr.update(choices=choices, value=[])
                except Exception as e:
                    return gr.update(choices=[f"Error: {e}"], value=[])

            def on_publish(selected, doctor_id):
                if not selected:
                    return "No alerts selected."
                try:
                    publish_alerts(selected, doctor_id)
                    return f"✅ Published {len(selected)} alert(s) to all doctors."
                except Exception as e:
                    return f"❌ Error: {e}"

            reg_refresh.click(
                fn=refresh_regional,
                inputs=[doctor_id_state],
                outputs=[reg_today, reg_week, reg_symptom_chart, reg_lang_chart,
                         reg_volume_chart, reg_doctor_chart]
            )
            gen_alerts_btn.click(fn=load_candidate_alerts, outputs=[alert_choices])
            publish_btn.click(
                fn=on_publish,
                inputs=[alert_choices, doctor_id_state],
                outputs=[publish_status]
            )

        # ── Tab 4: Health Alerts ──────────────────────────────────────────────
        with gr.Tab("🔔 Health Alerts"):
            gr.Markdown("*Alerts published by the head of department. Visible to all doctors.*")
            alerts_refresh = gr.Button("🔄 Load Alerts")
            alerts_html    = gr.HTML("<em>Click refresh to load alerts.</em>")

            def load_published_alerts():
                try:
                    alerts = fetch_published_alerts()
                    if not alerts:
                        return "<em>No alerts published yet.</em>"
                    cards = ""
                    colors = {"INFO":"#ebf8ff","WARN":"#fffaf0","CRITICAL":"#fff5f5"}
                    borders = {"INFO":"#90cdf4","WARN":"#fbd38d","CRITICAL":"#fc8181"}
                    for a in alerts:
                        sev = a.get("severity","INFO")
                        bg  = colors.get(sev,"#ebf8ff")
                        bd  = borders.get(sev,"#90cdf4")
                        cards += f"""
                        <div style="background:{bg};border-left:4px solid {bd};
                             border-radius:8px;padding:14px 16px;margin-bottom:10px">
                          <div style="font-weight:600;color:#2d3748;margin-bottom:4px">
                            {'⚠️' if sev=='WARN' else '🔴' if sev=='CRITICAL' else 'ℹ️'} {sev}
                          </div>
                          <div style="color:#4a5568;font-size:14px">{a['insight_text']}</div>
                          <div style="color:#a0aec0;font-size:11px;margin-top:6px">
                            Published by {a['published_by']} · {str(a['published_at'])[:16]}
                          </div>
                        </div>"""
                    return cards
                except Exception as e:
                    return f"<p style='color:red'>Error: {e}</p>"

            alerts_refresh.click(fn=load_published_alerts, outputs=[alerts_html])

        # ── Tab 5: Records Viewer ─────────────────────────────────────────────
        with gr.Tab("🗂️ Records Viewer"):

            gr.Markdown("Browse SOAP notes day-wise or patient-wise. Leave a filter blank to ignore it.")

            with gr.Row():
                filter_date = gr.Textbox(
                    label="📅 Date (YYYY-MM-DD) — leave blank for all dates",
                    placeholder="e.g. 2026-03-29",
                    scale=2,
                )
                filter_patient = gr.Textbox(
                    label="👤 Patient ID — leave blank for all patients",
                    placeholder="e.g. PAT9999",
                    scale=2,
                )
                filter_scope = gr.Radio(
                    choices=[("My records only", "mine"), ("All doctors", "all")],
                    value="mine",
                    label="👨‍⚕️ Scope",
                    scale=1,
                )

            with gr.Row():
                viewer_search_btn = gr.Button("🔍 Search", variant="primary", scale=1)
                viewer_today_btn  = gr.Button("📅 Today",  variant="secondary", scale=1)
                viewer_all_btn    = gr.Button("👤 This Patient (all dates)", variant="secondary", scale=1)

            viewer_count = gr.Markdown("")
            viewer_html  = gr.HTML(
                value="<div style='padding:20px;color:var(--body-text-color-subdued);text-align:center'>"
                      "Use the filters above and click Search.</div>"
            )

            def do_search(date_str, patient_id, scope, doctor_id, progress=gr.Progress()):
                progress(0, desc="Querying records...")
                records = fetch_records_filtered(date_str, patient_id, doctor_id, scope)
                progress(0.7, desc="Rendering...")
                html = render_records_html(records)
                progress(1.0, desc="Done")
                return html

            def do_today(scope, doctor_id, progress=gr.Progress()):
                from datetime import date
                today = date.today().isoformat()
                progress(0, desc="Loading today's records...")
                records = fetch_records_filtered(today, "", doctor_id, scope)
                progress(0.7, desc="Rendering...")
                html = render_records_html(records)
                progress(1.0)
                return today, html

            def do_patient_all(patient_id, scope, doctor_id, progress=gr.Progress()):
                if not patient_id.strip():
                    return "", "<div style='color:orange;padding:12px'>Enter a Patient ID first.</div>"
                progress(0, desc=f"Loading all visits for {patient_id}...")
                records = fetch_records_filtered("", patient_id, doctor_id, scope)
                progress(0.7, desc="Rendering...")
                html = render_records_html(records)
                progress(1.0)
                return "", html

            viewer_search_btn.click(
                fn=do_search,
                inputs=[filter_date, filter_patient, filter_scope, doctor_id_state],
                outputs=[viewer_html],
                show_progress="full",
            )
            viewer_today_btn.click(
                fn=do_today,
                inputs=[filter_scope, doctor_id_state],
                outputs=[filter_date, viewer_html],
                show_progress="full",
            )
            viewer_all_btn.click(
                fn=do_patient_all,
                inputs=[filter_patient, filter_scope, doctor_id_state],
                outputs=[filter_date, viewer_html],
                show_progress="full",
            )

            # Auto-search when patient ID is typed (with a small debounce via submit)
            filter_patient.submit(
                fn=do_search,
                inputs=[filter_date, filter_patient, filter_scope, doctor_id_state],
                outputs=[viewer_html],
                show_progress="full",
            )
# ── Tab 6: ML Analytics ───────────────────────────────────────────────
        with gr.Tab("🧬 ML Analytics"):

            gr.Markdown(
                "### Symptom Clustering & Anomaly Detection\n"
                "K-Means clustering on symptom vectors · "
                "Anomaly detection via centroid distance · "
                "Temporal spike detection"
            )

            with gr.Row():
                k_slider = gr.Slider(
                    minimum=2, maximum=8, step=1, value=4,
                    label="Max clusters to try (best K auto-selected by silhouette score)",
                    scale=3
                )
                ml_scope = gr.Radio(
                    choices=[("My patients", "mine"), ("All doctors", "all")],
                    value="all",
                    label="Scope",
                    scale=1
                )

            run_ml_btn = gr.Button("⚙️ Run ML Analysis", variant="primary")
            ml_summary = gr.Markdown("")

            # ── Charts row 1 ──────────────────────────────────────────────────
            with gr.Row():
                plot_silhouette = gr.Plot(label="Silhouette Score by K")
                plot_pca        = gr.Plot(label="Cluster Map (PCA projection)")

            # ── Charts row 2 ──────────────────────────────────────────────────
            plot_clusters = gr.Plot(label="Symptom Profiles per Cluster")

            # ── Charts row 3 ──────────────────────────────────────────────────
            with gr.Row():
                plot_anomaly_dist = gr.Plot(label="Anomaly Score Distribution")
                plot_temporal     = gr.Plot(label="Symptom Trends & Spikes")

            # ── Anomaly table ─────────────────────────────────────────────────
            gr.Markdown("### ⚠️ Flagged Anomalies")
            gr.Markdown(
                "*Records whose symptom combinations don't fit any cluster — "
                "unusual presentations worth reviewing.*"
            )
            anomaly_table = gr.Dataframe(
                headers=["Patient ID", "Doctor", "Symptoms", "Diagnosis",
                         "Anomaly Score", "Cluster"],
                datatype=["str","str","str","str","str","str"],
                row_count=(1, "dynamic"),
                interactive=False,
                elem_classes=["dark-table"],
            )

            def on_run_ml(k, scope, doctor_id, progress=gr.Progress()):
                empty = ("", None, None, None, None, None, [])
                try:
                    progress(0, desc="Fetching records from Delta Lake...")
                    records = fetch_all_records_for_ml()

                    # Filter by scope
                    if scope == "mine":
                        records = [r for r in records if r.get("doctor_id") == doctor_id]

                    if len(records) < 4:
                        return (
                            "⚠️ Not enough records. Add more patient data first.",
                            None, None, None, None, None, []
                        )

                    progress(0.2, desc=f"Running K-Means (trying K=2 to {k})...")
                    results, err = run_ml_analysis(records, n_clusters=k)

                    if err:
                        return (err, None, None, None, None, None, [])

                    progress(0.85, desc="Rendering charts...")
                    return (
                        results["summary"],
                        results["fig_silhouette"],
                        results["fig_pca"],
                        results["fig_clusters"],
                        results["fig_anomaly_dist"],
                        results["fig_temporal"],
                        results["anomaly_rows"],
                    )
                except Exception as e:
                    logging.exception("ML analysis error")
                    return (f"❌ Error: {e}", None, None, None, None, None, [])

            run_ml_btn.click(
                fn=on_run_ml,
                inputs=[k_slider, ml_scope, doctor_id_state],
                outputs=[
                    ml_summary,
                    plot_silhouette,
                    plot_pca,
                    plot_clusters,
                    plot_anomaly_dist,
                    plot_temporal,
                    anomaly_table,
                ],
                show_progress="full",
            )
    return demo

def main():
    logging.basicConfig(level="INFO")
    _load_secrets()
    demo = build_app()
    demo.queue()
    demo.launch()  # bare launch — platform injects all env vars

if __name__ == "__main__":
    main()




