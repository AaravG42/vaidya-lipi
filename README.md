# Vaidya Lipi (वैद्य लिपि)

Vaidya Lipi is an AI-powered clinical scribe that converts multilingual doctor-patient consultations into structured, ABDM-compliant medical records. It automates note-taking by extracting symptoms, medications, and SNOMED-CT clinical entities in real-time.

## Architecture

Vaidya Lipi uses a hybrid pipeline combining multilingual ASR, clinical-specific embeddings, and large language models, all orchestrated on the Databricks Lakehouse platform.

```mermaid
graph TD
    A[Gradio UI (Databricks Apps)] -->|Audio| B[Sarvam Saaras v3 (Speech-to-Text)]
    B -->|Transcript| C[Databricks Serving (Llama 3.3 70B)]
    B -->|Transcript| D[Parrotlet-e + FAISS (SNOMED-CT Mapping)]
    C -->|Structured SOAP Note| E[Databricks SQL Warehouse]
    D -->|Clinical Entities| E
    E -->|Storage| F[Delta Lake (Unity Catalog)]
    E -->|Analytics| G[Doctor Dashboard & Alerts]
```

## How to Run

### 1. Prerequisites
- **Databricks Workspace** with Unity Catalog enabled.
- **Sarvam AI API Key** for multilingual transcription.
- **HuggingFace Token** to download the `ekacare/parrotlet-e` model.

### 2. Configure Secrets
Use the Databricks CLI to store your API keys securely in a secret scope named `vaidya-lipi`:

```bash
databricks secrets create-scope vaidya-lipi
databricks secrets put-secret vaidya-lipi sarvam_api_key
databricks secrets put-secret vaidya-lipi hf_token
```

### 3. Initialize Database
Run the `notebooks/01_ingest.ipynb` notebook to create the Unity Catalog schema and Delta Lake tables:
- `workspace.vaidya.patient_records`
- `workspace.vaidya.health_alerts`

### 4. Deploy Application
1. Connect this repository to your Databricks workspace via Git.
2. Navigate to **Compute > Apps > Create App**.
3. Select this repository and set the entry file to `app/main.py`.
4. Ensure `app.yaml` is in the root directory for environment configuration.
5. Deploy the application.

## Demo Steps

1. **Patient Intake**: Open the application and go to the **Record Consultation** tab. Enter a **Patient ID** (e.g., `PAT-1234`).
2. **Consultation**: Click the microphone in the **Audio Recording** section. Speak in Hindi, English, or any supported Indian language (e.g., *"Patient has high fever and severe headache for two days"*).
3. **Transcription**: Click **↺ Transcribe Audio**. The multilingual transcript will appear in the box.
4. **Processing**: Click **✦ Structure & Save Record**. The system will:
   - Generate a structured **SOAP Note** (Subjective, Objective, Assessment, Plan).
   - Map symptoms to **SNOMED-CT** clinical codes.
   - Save the record to Delta Lake.
5. **Dashboard**: Switch to the **Dashboard** tab to view the daily summary of patients seen, top symptoms, and language breakdown.
6. **Alerts**: Navigate to the **Health Alerts** tab to see population-level insights generated from Spark analytics.
