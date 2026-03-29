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
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="teal"),
        css=CSS,
        title="Vaidya Lipi — Medical Scribe"
    ) as demo:

        DOCTOR_ID = "DR001"   # in production this comes from login

        gr.Markdown(
            "# Vaidya Lipi · वैद्य लिपि\n"
            "*AI Medical Scribe · ABDM-compatible · "
            f"Logged in as **{DOCTOR_ID}** (Head of Department)*"
        )

        doctor_id_state = gr.State(DOCTOR_ID)
        # Store structured result between Process and Save steps
        structured_state = gr.State({})
        entities_state   = gr.State([])

        # ── Tab 1: Record Consultation ────────────────────────────────────────
        with gr.Tab("📝 Record Consultation"):

            with gr.Row():
                patient_id_box = gr.Textbox(
                    label="Patient ID (ABHA ID or local ID)",
                    placeholder="e.g. PAT1234",
                    scale=3
                )
                doctor_id_box = gr.Textbox(
                    label="Doctor ID",
                    value=DOCTOR_ID,
                    scale=1
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
                transcribe_btn = gr.Button("🎙️ Transcribe Audio", variant="secondary", scale=1)
                process_btn    = gr.Button("⚙️  Analyse Transcript", variant="secondary", scale=1)
                save_btn       = gr.Button("💾  Save Record", variant="primary", scale=1, interactive=False)

            # Results area — hidden until Process is clicked
            results_html = gr.HTML(visible=False)
            save_status  = gr.Textbox(label="Status", interactive=False, visible=False)

            # ── Handlers ──────────────────────────────────────────────────────
            def on_transcribe(audio):
                if audio is None:
                    return "No audio recorded."
                try:
                    transcript, lang = transcribe_audio(audio)
                    return f"[{lang}] {transcript}"
                except Exception as e:
                    return f"Transcription error: {e}"

            def on_process(transcript, progress=gr.Progress()):
                if not transcript.strip():
                    return (
                        gr.update(visible=False),
                        {},
                        [],
                        gr.update(interactive=False),
                        gr.update(visible=False),
                    )
                try:
                    structured = structure_transcript(transcript)
                    entities   = extract_medical_entities(transcript)
                    html       = render_soap_html(structured, entities)
                    return (
                        gr.update(value=html, visible=True),
                        structured,
                        entities,
                        gr.update(interactive=True),   # enable Save button
                        gr.update(visible=False),
                    )
                except Exception as e:
                    logging.exception("process error")
                    return (
                        gr.update(value=f"<p style='color:red'>Error: {e}</p>", visible=True),
                        {},
                        [],
                        gr.update(interactive=False),
                        gr.update(visible=False),
                    )

            def on_save(patient_id, doctor_id, transcript, structured, entities):
                if not structured:
                    return gr.update(value="Nothing to save — process first.", visible=True)
                try:
                    record_id = save_record(
                        patient_id=patient_id or "UNKNOWN",
                        doctor_id=doctor_id or DOCTOR_ID,
                        transcript=transcript,
                        structured=structured,
                        entities=entities,
                        language="mixed"
                    )
                    return gr.update(
                        value=f"✅ Saved successfully. Record ID: `{record_id}`",
                        visible=True
                    )
                except Exception as e:
                    logging.exception("save error")
                    return gr.update(value=f"❌ Save failed: {e}", visible=True)

            transcribe_btn.click(
                fn=on_transcribe,
                inputs=[audio_input],
                outputs=[transcript_box]
            )
            process_btn.click(
                fn=on_process,
                inputs=[transcript_box],
                outputs=[results_html, structured_state, entities_state, save_btn, save_status]
            )
            save_btn.click(
                fn=on_save,
                inputs=[patient_id_box, doctor_id_box, transcript_box,
                        structured_state, entities_state],
                outputs=[save_status]
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

    return demo

def main():
    logging.basicConfig(level="INFO")
    _load_secrets()
    demo = build_app()
    demo.queue()
    demo.launch()  # bare launch — platform injects all env vars

if __name__ == "__main__":
    main()




