"""
ShieldTab backend — FastAPI server exposing /predict for the Chrome extension.
Loads the XGBoost + CNN ensemble once at startup for low-latency inference.
"""
import os
import sys
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))
from ensemble import get_ensemble  # noqa: E402
from rag_explain import get_explainer  # noqa: E402

app = FastAPI(title="ShieldTab API", version="0.1.0")

# Chrome extensions call this from a chrome-extension:// origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ensemble = None
explainer = None


@app.on_event("startup")
def load_models():
    global ensemble, explainer
    print("Loading ShieldTab ensemble models...")
    ensemble = get_ensemble()
    print("Loading RAG explainer...")
    explainer = get_explainer()
    print("Models loaded.")


class PredictRequest(BaseModel):
    url: str


class ExplainRequest(BaseModel):
    url: str
    threat_type: str = "malicious"


class SimilarThreat(BaseModel):
    url: str
    type: str
    similarity: float


class ExplainResponse(BaseModel):
    summary: str
    similar_threats: list[SimilarThreat]
    signals: list[str]
    latency_ms: float


class PredictResponse(BaseModel):
    url: str
    is_malicious: bool
    confidence: float
    threat_type: str
    xgb_score: float | None
    cnn_score: float | None
    latency_ms: float
    trusted_domain: bool = False


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": ensemble is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.url or not req.url.strip():
        raise HTTPException(status_code=400, detail="Empty URL")

    t0 = time.time()
    result = ensemble.predict(req.url.strip())
    result["latency_ms"] = round((time.time() - t0) * 1000, 2)
    return result


@app.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest):
    """Separate from /predict so the blocking path stays fast -- this is
    called on-demand (e.g. a "why was this flagged?" button) rather than
    on every navigation."""
    if not req.url or not req.url.strip():
        raise HTTPException(status_code=400, detail="Empty URL")
    if explainer is None or not explainer.available:
        raise HTTPException(status_code=503, detail="Explainer not available")

    t0 = time.time()
    result = explainer.explain(req.url.strip(), req.threat_type)
    result["latency_ms"] = round((time.time() - t0) * 1000, 2)
    return result


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # deploy hosts (Render, etc.) set PORT
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
