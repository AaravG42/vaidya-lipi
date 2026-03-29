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
            vol = "/Volumes/main/vaidya/models_and_indexes/parrotlet_index"
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

def structure_transcript(transcript: str) -> dict:
    """Call Llama 4 Maverick via AI Gateway to structure the transcript."""
    import requests

    base_url = os.environ.get("LLM_OPENAI_BASE_URL", "")
    model = os.environ.get("LLM_MODEL", "databricks-llama-4-maverick")

    if not base_url:
        raise ValueError("LLM_OPENAI_BASE_URL not set in app.yaml")

    # Get OAuth token (works on Databricks Apps, no PAT needed)
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    headers_auth = w.config.authenticate()

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SOAP_PROMPT},
            {"role": "user", "content": transcript}
        ],
        "max_tokens": 1024,
        "temperature": 0.1,  # low temp for structured extraction
    }

    resp = requests.post(
        f"{base_url}/chat/completions",
        headers={**headers_auth, "Content-Type": "application/json"},
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]

    # Strip markdown fences if present
    content = content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Fallback: return raw text in a structured wrapper
        return {
            "symptoms": [], "medications": [], "diagnosis": "See raw note",
            "plan": content, "soap_s": transcript,
            "soap_o": "", "soap_a": "", "soap_p": ""
        }


def save_record(patient_id: str, doctor_id: str, transcript: str,
                structured: dict, entities: list, language: str) -> str:
    """Write a completed consultation record to Delta Lake."""
    from pyspark.sql import SparkSession
    from pyspark.sql import Row
    from datetime import datetime

    spark = SparkSession.builder.getOrCreate()
    record_id = str(uuid.uuid4())

    row = Row(
        record_id=record_id,
        patient_id=patient_id,
        doctor_id=doctor_id,
        hospital_id=os.environ.get("HOSPITAL_ID", "DEMO_HOSPITAL"),
        timestamp=datetime.now(),
        language_detected=language,
        raw_transcript=transcript,
        symptoms=structured.get("symptoms", []),
        diagnosis=structured.get("diagnosis"),
        medications=structured.get("medications", []),
        snomed_codes=[e["concept_id"] for e in entities],
        structured_note=json.dumps(structured),
        soap_subjective=structured.get("soap_s"),
        soap_objective=structured.get("soap_o"),
        soap_assessment=structured.get("soap_a"),
        soap_plan=structured.get("soap_p"),
        is_anonymized=False,
    )

    df = spark.createDataFrame([row])
    df.write.mode("append").saveAsTable("workspace.vaidya.patient_records")
    return record_id


def get_doctor_dashboard(doctor_id: str) -> dict:
    """PySpark aggregations on Delta Lake for the doctor's dashboard."""
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    spark = SparkSession.builder.getOrCreate()

    df = spark.sql(f"""
        SELECT * FROM workspace.vaidya.patient_records
        WHERE doctor_id = '{doctor_id}'
        AND DATE(timestamp) = CURRENT_DATE()
    """)

    total = df.count()

    # Explode symptoms array and count
    symptom_counts = (
        df.select(F.explode("symptoms").alias("symptom"))
        .groupBy("symptom").count()
        .orderBy("count", ascending=False)
        .limit(5)
        .collect()
    )
    top_symptoms = [(row["symptom"], row["count"]) for row in symptom_counts]

    # Language breakdown
    lang_counts = (
        df.groupBy("language_detected").count()
        .collect()
    )
    languages = {row["language_detected"]: row["count"] for row in lang_counts}

    return {
        "total_patients_today": total,
        "top_symptoms": top_symptoms,
        "languages": languages,
    }

def build_app() -> gr.Blocks:
    with gr.Blocks(
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="teal"),
        title="Vaidya Lipi — Medical Scribe"
    ) as demo:

        gr.Markdown("# Vaidya Lipi · वैद्य लिपि\n*AI Medical Scribe · ABDM-compatible*")

        # State lives OUTSIDE tabs so both tabs can share it
        doctor_id_state = gr.State("DR001")

        with gr.Tab("Record Consultation"):
            with gr.Row():
                patient_id_box = gr.Textbox(
                    label="Patient ID (ABHA ID or local ID)",
                    placeholder="e.g. PAT1234 or 14-digit ABHA",
                    scale=2
                )
                doctor_id_box = gr.Textbox(
                    label="Doctor ID",
                    value="DR001",
                    scale=1
                )

            # Sync the textbox into State whenever it changes
            doctor_id_box.change(
                fn=lambda x: x,
                inputs=[doctor_id_box],
                outputs=[doctor_id_state]
            )

            gr.Markdown("**Step 1:** Enter patient ID. **Step 2:** Record. **Step 3:** Click Transcribe.")

            audio_input = gr.Audio(
                sources=["microphone"],
                type="numpy",
                label="Record consultation (Hindi, English, or any Indian language)",
            )

            transcript_box = gr.Textbox(
                label="Transcript (auto-filled, edit if needed)",
                lines=4,
                placeholder="Transcript will appear here after recording..."
            )

            with gr.Row():
                transcribe_btn = gr.Button("Transcribe Audio", variant="secondary")
                process_btn = gr.Button("Structure & Save Record", variant="primary")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### SOAP Note")
                    soap_box = gr.JSON(label="Structured Note")
                with gr.Column():
                    gr.Markdown("### Detected Medical Entities (SNOMED)")
                    entities_box = gr.JSON(label="Entities (Parrotlet-e)")

            status_box = gr.Textbox(label="Status", interactive=False)

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
                        language="mixed"
                    )
                    return structured, entities, f"✓ Saved. Record ID: {record_id}"
                except Exception as e:
                    logging.exception("process error")
                    return {}, [], f"Error: {e}"

            # Only button click — no stop_recording
            transcribe_btn.click(
                fn=on_transcribe,
                inputs=[audio_input],
                outputs=[transcript_box]
            )
            process_btn.click(
                fn=on_process,
                inputs=[patient_id_box, doctor_id_box, transcript_box],
                outputs=[soap_box, entities_box, status_box]
            )

        with gr.Tab("Doctor Dashboard"):
            refresh_btn = gr.Button("Refresh Dashboard", variant="secondary")
            total_box = gr.Number(label="Patients Seen Today")
            symptoms_box = gr.JSON(label="Top 5 Symptoms Today")
            lang_box = gr.JSON(label="Language Breakdown")

            def refresh_dashboard(doctor_id):
                try:
                    data = get_doctor_dashboard(doctor_id)
                    return (
                        data["total_patients_today"],
                        data["top_symptoms"],
                        data["languages"]
                    )
                except Exception as e:
                    return 0, [], {"error": str(e)}

            # Use State, not the textbox from Tab 1
            refresh_btn.click(
                fn=refresh_dashboard,
                inputs=[doctor_id_state],
                outputs=[total_box, symptoms_box, lang_box]
            )

        with gr.Tab("Health Alerts"):
            gr.Markdown("### Population-level insights from Spark analytics")
            gr.Markdown("*(Run Notebook 03 to generate alerts, then refresh)*")
            alerts_refresh = gr.Button("Load Alerts")
            alerts_display = gr.JSON(label="Current Alerts")

            def load_alerts():
                from pyspark.sql import SparkSession
                spark = SparkSession.builder.getOrCreate()
                try:
                    df = spark.sql("""
                        SELECT * FROM workspace.vaidya.health_alerts
                        ORDER BY generated_at DESC LIMIT 5
                    """)
                    return [row.asDict() for row in df.collect()]
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





