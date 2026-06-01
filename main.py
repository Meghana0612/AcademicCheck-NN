"""
backend/main.py
----------------
FastAPI backend for the Plagiarism Detector.

Endpoints:
  POST /api/analyse          → analyse two text inputs
  POST /api/upload           → upload .txt or .pdf file
  GET  /api/history          → get past check history
  DELETE /api/history/{id}   → delete a history entry
  GET  /api/health           → health check
"""

import os, uuid, json
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Internal modules
from src.predictor import PlagiarismPredictor

app = FastAPI(title="Plagiarism Detector API", version="2.0.0")

# Allow React frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
predictor = PlagiarismPredictor()

# In-memory history store (resets on server restart)
history_store: list = []

# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class AnalyseRequest(BaseModel):
    text1: str
    text2: str

class AnalyseResponse(BaseModel):
    id: str
    verdict: str
    probability: float
    tfidf_similarity: float
    embedding_similarity: float
    length_similarity: float
    shap_values: dict
    lime_weights: list
    timestamp: str

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": predictor.is_ready()}


@app.post("/api/analyse", response_model=AnalyseResponse)
def analyse(req: AnalyseRequest):
    """Analyse two texts for plagiarism."""
    if not req.text1.strip() or not req.text2.strip():
        raise HTTPException(status_code=400, detail="Both texts must be non-empty.")

    result = predictor.predict(req.text1, req.text2)
    result["id"] = str(uuid.uuid4())
    result["timestamp"] = datetime.now().isoformat()

    # Save to history
    history_store.append({
        "id":          result["id"],
        "text1":       req.text1[:120] + ("…" if len(req.text1) > 120 else ""),
        "text2":       req.text2[:120] + ("…" if len(req.text2) > 120 else ""),
        "verdict":     result["verdict"],
        "probability": result["probability"],
        "timestamp":   result["timestamp"],
    })

    return result


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Accept a .txt or .pdf file and return its extracted text.
    The frontend then places the text into the text box.
    """
    filename = file.filename.lower()

    if filename.endswith(".txt"):
        content = await file.read()
        text = content.decode("utf-8", errors="ignore")

    elif filename.endswith(".pdf"):
        try:
            import pdfplumber
            content = await file.read()
            import io
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                text = "\n".join(
                    page.extract_text() or "" for page in pdf.pages
                )
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="pdfplumber not installed. Run: pip install pdfplumber"
            )
    else:
        raise HTTPException(
            status_code=400,
            detail="Only .txt and .pdf files are supported."
        )

    return {"filename": file.filename, "text": text.strip(), "char_count": len(text)}


@app.get("/api/history")
def get_history():
    """Return all past plagiarism checks, newest first."""
    return list(reversed(history_store))


@app.delete("/api/history/{entry_id}")
def delete_history(entry_id: str):
    """Delete a specific history entry by ID."""
    global history_store
    before = len(history_store)
    history_store = [h for h in history_store if h["id"] != entry_id]
    if len(history_store) == before:
        raise HTTPException(status_code=404, detail="Entry not found.")
    return {"deleted": entry_id}
