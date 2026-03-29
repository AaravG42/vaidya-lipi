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

_CUSTOM_CSS = """
/* ── Global ── */
:root {
    --vl-primary: #2563eb;
    --vl-primary-hover: #1d4ed8;
    --vl-accent: #0d9488;
    --vl-bg: #f8fafc;
    --vl-surface: #ffffff;
    --vl-border: #e2e8f0;
    --vl-text: #1e293b;
    --vl-text-muted: #64748b;
    --vl-success: #059669;
    --vl-radius: 12px;
}
.gradio-container {
    max-width: 1120px !important;
    margin: 0 auto !important;
    background: var(--vl-bg) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}
footer { display: none !important; }

/* ── Header ── */
.app-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
    border-radius: var(--vl-radius);
    padding: 28px 32px;
    margin-bottom: 24px;
    color: white;
    text-align: center;
}
.app-header h1 {
    margin: 0 0 4px 0;
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: -0.02em;
}
.app-header p {
    margin: 0;
    opacity: 0.85;
    font-size: 0.9rem;
    font-weight: 400;
}

/* ── Cards ── */
.card {
    background: var(--vl-surface);
    border: 1px solid var(--vl-border);
    border-radius: var(--vl-radius);
    padding: 24px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    transition: box-shadow 0.2s ease;
}
.card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.06); }

/* ── Step indicator ── */
.steps-bar {
    display: flex;
    gap: 8px;
    margin-bottom: 20px;
    flex-wrap: wrap;
}
.step-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #eff6ff;
    color: var(--vl-primary);
    border: 1px solid #bfdbfe;
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 0.8rem;
    font-weight: 500;
}
.step-chip .num {
    background: var(--vl-primary);
    color: white;
    border-radius: 50%;
    width: 20px;
    height: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.7rem;
    font-weight: 700;
}

/* ── Section headers ── */
.section-title {
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--vl-text-muted);
    margin: 0 0 12px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid var(--vl-border);
}

/* ── Buttons ── */
.btn-primary button {
    background: var(--vl-primary) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 10px 28px !important;
    font-size: 0.9rem !important;
    transition: all 0.15s ease !important;
}
.btn-primary button:hover {
    background: var(--vl-primary-hover) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(37,99,235,0.3) !important;
}
.btn-secondary button {
    background: var(--vl-surface) !important;
    border: 1.5px solid var(--vl-border) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    color: var(--vl-text) !important;
    padding: 10px 28px !important;
    font-size: 0.9rem !important;
    transition: all 0.15s ease !important;
}
.btn-secondary button:hover {
    background: #f1f5f9 !important;
    border-color: #cbd5e1 !important;
}

/* ── Status bar ── */
.status-bar textarea, .status-bar input {
    background: #f0fdf4 !important;
    border: 1px solid #bbf7d0 !important;
    border-radius: 8px !important;
    color: var(--vl-success) !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
}

/* ── Dashboard stat card ── */
.stat-card {
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    border: 1px solid #bfdbfe;
    border-radius: var(--vl-radius);
    padding: 20px 24px;
    text-align: center;
}
.stat-card .stat-value {
    font-size: 2.5rem;
    font-weight: 800;
    color: var(--vl-primary);
    line-height: 1;
    margin-bottom: 4px;
}
.stat-card .stat-label {
    font-size: 0.8rem;
    color: var(--vl-text-muted);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Alert badge ── */
.alert-info {
    background: #fefce8;
    border: 1px solid #fde68a;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.85rem;
    color: #92400e;
    margin-bottom: 12px;
}

/* ── Tabs ── */
.tabs > .tab-nav > button {
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 10px 20px !important;
    border-radius: 8px 8px 0 0 !important;
}
.tabs > .tab-nav > button.selected {
    color: var(--vl-primary) !important;
    border-bottom: 2px solid var(--vl-primary) !important;
}

/* ── Inputs ── */
.gradio-container input, .gradio-container textarea {
    border-radius: 8px !important;
    border: 1.5px solid var(--vl-border) !important;
    transition: border-color 0.15s ease !important;
}
.gradio-container input:focus, .gradio-container textarea:focus {
    border-color: var(--vl-primary) !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important;
}

/* ── JSON display ── */
.json-holder {
    border-radius: var(--vl-radius) !important;
    border: 1px solid var(--vl-border) !important;
}

/* ── Responsive ── */
@media (max-width: 768px) {
    .gradio-container { padding: 8px !important; }
    .app-header { padding: 20px 16px; }
    .app-header h1 { font-size: 1.35rem; }
    .card { padding: 16px; }
    .steps-bar { flex-direction: column; }
}
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="teal",
            neutral_hue="slate",
            font=("Inter", "system-ui", "sans-serif"),
            radius_size=gr.themes.sizes.radius_md,
        ),
        css=_CUSTOM_CSS,
        title="Vaidya Lipi — AI Medical Scribe",
    ) as demo:

        # ── Header ──
        gr.HTML("""
            <div class="app-header">
                <h1>Vaidya Lipi &middot; &#x0935;&#x0948;&#x0926;&#x094D;&#x092F; &#x0932;&#x093F;&#x092A;&#x093F;</h1>
                <p>AI-Powered Medical Scribe &middot; ABDM-Compatible &middot; Multilingual</p>
            </div>
        """)

        # State lives OUTSIDE tabs so both tabs can share it
        doctor_id_state = gr.State("DR001")

        # ═══════════════════ Tab 1: Record Consultation ═══════════════════
        with gr.Tab("Record Consultation"):

            # ── Patient & Doctor IDs ──
            with gr.Group(elem_classes="card"):
                gr.HTML('<p class="section-title">Patient & Doctor Info</p>')
                with gr.Row(equal_height=True):
                    patient_id_box = gr.Textbox(
                        label="Patient ID",
                        placeholder="e.g. PAT1234 or 14-digit ABHA",
                        scale=2,
                    )
                    doctor_id_box = gr.Textbox(
                        label="Doctor ID",
                        value="DR001",
                        scale=1,
                    )

            # Sync the textbox into State whenever it changes
            doctor_id_box.change(
                fn=lambda x: x,
                inputs=[doctor_id_box],
                outputs=[doctor_id_state],
            )

            # ── Workflow steps ──
            gr.HTML("""
                <div class="steps-bar">
                    <span class="step-chip"><span class="num">1</span> Enter patient ID</span>
                    <span class="step-chip"><span class="num">2</span> Record audio</span>
                    <span class="step-chip"><span class="num">3</span> Transcribe</span>
                    <span class="step-chip"><span class="num">4</span> Structure &amp; save</span>
                </div>
            """)

            # ── Audio + Transcript ──
            with gr.Group(elem_classes="card"):
                gr.HTML('<p class="section-title">Consultation Recording</p>')
                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    label="Record (Hindi, English, or any Indian language)",
                )
                transcript_box = gr.Textbox(
                    label="Transcript",
                    lines=4,
                    placeholder="Transcript will appear here after recording — you can edit before saving.",
                )
                with gr.Row():
                    transcribe_btn = gr.Button(
                        "Transcribe Audio",
                        variant="secondary",
                        elem_classes="btn-secondary",
                    )
                    process_btn = gr.Button(
                        "Structure & Save Record",
                        variant="primary",
                        elem_classes="btn-primary",
                    )

            # ── Results ──
            with gr.Row(equal_height=True):
                with gr.Column(scale=3):
                    with gr.Group(elem_classes="card"):
                        gr.HTML('<p class="section-title">SOAP Note</p>')
                        soap_box = gr.JSON(label="Structured Note")
                with gr.Column(scale=2):
                    with gr.Group(elem_classes="card"):
                        gr.HTML('<p class="section-title">Medical Entities (SNOMED)</p>')
                        entities_box = gr.JSON(label="Entities (Parrotlet-e)")

            # ── Status ──
            status_box = gr.Textbox(
                label="Status",
                interactive=False,
                elem_classes="status-bar",
            )

            # ── Event handlers ──
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
                    return structured, entities, f"Saved. Record ID: {record_id}"
                except Exception as e:
                    logging.exception("process error")
                    return {}, [], f"Error: {e}"

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

        # ═══════════════════ Tab 2: Doctor Dashboard ═══════════════════
        with gr.Tab("Dashboard"):
            with gr.Group(elem_classes="card"):
                gr.HTML('<p class="section-title">Today\'s Overview</p>')
                refresh_btn = gr.Button(
                    "Refresh Dashboard",
                    variant="secondary",
                    elem_classes="btn-secondary",
                )

            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    with gr.Group(elem_classes="card"):
                        total_box = gr.Number(label="Patients Seen Today")
                with gr.Column(scale=1):
                    with gr.Group(elem_classes="card"):
                        lang_box = gr.JSON(label="Language Breakdown")
                with gr.Column(scale=1):
                    with gr.Group(elem_classes="card"):
                        symptoms_box = gr.JSON(label="Top 5 Symptoms Today")

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

            refresh_btn.click(
                fn=refresh_dashboard,
                inputs=[doctor_id_state],
                outputs=[total_box, symptoms_box, lang_box],
            )

        # ═══════════════════ Tab 3: Health Alerts ═══════════════════
        with gr.Tab("Health Alerts"):
            with gr.Group(elem_classes="card"):
                gr.HTML('<p class="section-title">Population-Level Insights</p>')
                gr.HTML(
                    '<div class="alert-info">Run Notebook 03 to generate alerts, then refresh below.</div>'
                )
                alerts_refresh = gr.Button(
                    "Load Alerts",
                    variant="secondary",
                    elem_classes="btn-secondary",
                )
                alerts_display = gr.JSON(label="Current Alerts")

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
    demo.launch()

if __name__ == "__main__":
    main()





