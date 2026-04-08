"""
main.py
-------
SpamShield API — FastAPI backend with real trained sklearn models.

Startup:
    1. Run train.py once to generate models/ directory
    2. uvicorn main:app --reload
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SpamShield API",
    description="SMS/email spam classifier — Naive Bayes · Logistic Regression · Ensemble",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Load models at startup
# ---------------------------------------------------------------------------

MODELS_DIR = Path("models")
_LOADED: dict = {}


def _load_models() -> None:
    """Load persisted sklearn pipelines + measured metrics from disk."""
    required = ["naive_bayes.pkl", "logistic_regression.pkl", "ensemble.pkl", "metrics.json"]
    missing = [f for f in required if not (MODELS_DIR / f).exists()]
    if missing:
        raise RuntimeError(
            f"Missing model files: {missing}. "
            "Run `python train.py` first to generate them."
        )

    with open(MODELS_DIR / "metrics.json") as f:
        metrics = json.load(f)

    _LOADED["naive_bayes"] = {
        "name": "Naive Bayes",
        "pipeline": joblib.load(MODELS_DIR / "naive_bayes.pkl"),
        **metrics["naive_bayes"],
    }
    _LOADED["logistic_regression"] = {
        "name": "Logistic Regression",
        "pipeline": joblib.load(MODELS_DIR / "logistic_regression.pkl"),
        **metrics["logistic_regression"],
    }
    # Ensemble is stored as {"nb": pipe, "lr": pipe, "weights": [...], "classes": [...]}
    _LOADED["ensemble"] = {
        "name": "Ensemble (NB + LR)",
        "pipeline": joblib.load(MODELS_DIR / "ensemble.pkl"),
        **metrics["ensemble"],
    }


@app.on_event("startup")
def startup() -> None:
    _load_models()


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

MAX_TEXT_LEN = 5_000


def _predict_single(pipeline, text: str) -> tuple[str, float]:
    """Return (label, spam_probability) for a single pipeline."""
    proba = pipeline.predict_proba([text])[0]   # shape: (2,)
    classes = list(pipeline.classes_)
    spam_idx = classes.index("spam")
    spam_prob = float(proba[spam_idx])
    label = "spam" if spam_prob >= 0.5 else "ham"
    return label, spam_prob


def _predict_ensemble(ens: dict, text: str) -> tuple[str, float]:
    """Soft-voting ensemble: weighted average of predict_proba."""
    nb_proba = ens["nb"].predict_proba([text])[0]
    lr_proba = ens["lr"].predict_proba([text])[0]
    w = ens["weights"]
    avg = w[0] * nb_proba + w[1] * lr_proba
    classes = ens["classes"]
    spam_idx = classes.index("spam")
    spam_prob = float(avg[spam_idx])
    label = "spam" if spam_prob >= 0.5 else "ham"
    return label, spam_prob


def _run_model(key: str, text: str) -> dict:
    """Unified inference entry point for any model key."""
    info = _LOADED[key]
    pipe = info["pipeline"]

    if key == "ensemble":
        label, spam_prob = _predict_ensemble(pipe, text)
    else:
        label, spam_prob = _predict_single(pipe, text)

    confidence = spam_prob if label == "spam" else 1.0 - spam_prob
    return {
        "label": label,
        "score": round(spam_prob * 100, 1),
        "confidence": round(confidence * 100, 1),
    }


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------

# Patterns reused only for explainability highlights — NOT for classification
_SPAM_HINTS = [
    (r"\bfree\b",                   "Common spam trigger word"),
    (r"\bwon\b|\bwinner\b",         "Prize/winner language"),
    (r"\bcongratulations?\b",       "Congratulations language"),
    (r"\bclaim\b",                  "Call-to-claim language"),
    (r"\burgent\b|\bact now\b",     "Urgency language"),
    (r"\blimited time\b",           "Scarcity language"),
    (r"\bclick here\b",             "Generic CTA"),
    (r"\bmillion\b",                "Large number reference"),
    (r"\blottery\b|\bjackpot\b",    "Lottery/jackpot language"),
    (r"\bcash\b|\$\d+",             "Cash/monetary reference"),
    (r"\bviagra\b|\bpills?\b",      "Pharmaceutical spam"),
    (r"\bverify.*account\b",        "Account verification request"),
    (r"\bpassword\b.*\bexpir",      "Password expiry threat"),
    (r"\bbank.*detail\b",           "Banking detail request"),
    (r"\b100%\b",                   "Percentage guarantee"),
    (r"\bno risk\b|\brisk.free\b",  "Risk-free claim"),
    (r"\bmake money\b",             "Money-making claim"),
    (r"\bwork from home\b",         "Work-from-home offer"),
    (r"!!!+",                       "Excessive exclamation marks"),
    (r"\$\$\$",                     "Dollar sign spam indicator"),
    (r"[A-Z]{5,}",                  "Excessive capitalisation"),
]

_HAM_HINTS = [
    (r"\bmeeting\b|\bschedule\b",   "Professional scheduling language"),
    (r"\bplease\b|\bkindly\b",      "Polite professional language"),
    (r"\breport\b|\bupdate\b",      "Business update language"),
    (r"\bteam\b|\bproject\b",       "Team/project context"),
    (r"\bthanks?\b|\bthank you\b",  "Gratitude — legitimate signal"),
    (r"\battached\b|\benclosed\b",  "Document attachment language"),
    (r"\bregards?\b|\bsincerely\b", "Professional sign-off"),
]

_SPAM_WORDS = {
    "free", "won", "winner", "congratulations", "claim", "urgent", "limited",
    "click", "million", "lottery", "jackpot", "cash", "prize", "verify",
    "account", "password", "bank", "risk", "earn", "make", "money", "home",
    "offer", "deal", "discount", "buy", "order", "guarantee",
}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-z]+\b", text.lower())


def _explain(text: str) -> list[dict]:
    lower = text.lower()
    seen: set[str] = set()
    out: list[dict] = []

    for pattern, reason in _SPAM_HINTS:
        m = re.search(pattern, lower, re.IGNORECASE)
        if m:
            word = m.group(0).strip()
            if word not in seen:
                seen.add(word)
                out.append({"word": word, "direction": "spam", "reason": reason})

    for pattern, reason in _HAM_HINTS:
        m = re.search(pattern, lower, re.IGNORECASE)
        if m:
            word = m.group(0).strip()
            if word not in seen:
                seen.add(word)
                out.append({"word": word, "direction": "ham", "reason": reason})

    freq = Counter(_tokenize(text))
    for word, count in freq.most_common(10):
        if word in _SPAM_WORDS and word not in seen:
            seen.add(word)
            out.append({
                "word": word,
                "direction": "spam",
                "reason": f"High-frequency spam word (×{count})",
            })

    return out[:8]


def _text_stats(text: str) -> dict:
    words = _tokenize(text)
    sentences = [s for s in re.split(r"[.!?]+", text.strip()) if s.strip()]
    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "caps_ratio": round(sum(c.isupper() for c in text) / max(len(text), 1) * 100, 1),
        "exclamation_count": text.count("!"),
        "url_count": len(re.findall(r"https?://\S+|www\.\S+", text)),
        "spam_word_count": sum(1 for w in words if w in _SPAM_WORDS),
    }


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LEN)
    model: Optional[str] = "ensemble"


class BatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=50)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/predict")
async def predict(req: PredictRequest):
    model_key = req.model if req.model in _LOADED else "ensemble"

    t0 = time.perf_counter()
    result = _run_model(model_key, req.text)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    # Compare all models (single inference each)
    all_models = {}
    for key, info in _LOADED.items():
        r = _run_model(key, req.text)
        all_models[key] = {
            "name":      info["name"],
            "label":     r["label"],
            "score":     r["score"],
            "accuracy":  info.get("accuracy"),
            "f1":        info.get("f1"),
            "precision": info.get("precision"),
            "recall":    info.get("recall"),
        }

    return {
        "label":        result["label"],
        "confidence":   result["confidence"],
        "score":        result["score"],
        "model_used":   _LOADED[model_key]["name"],
        "inference_ms": elapsed_ms,
        "highlights":   _explain(req.text),
        "stats":        _text_stats(req.text),
        "all_models":   all_models,
    }


@app.post("/batch")
async def batch_predict(req: BatchRequest):
    results = []
    for text in req.texts:
        if not text.strip():
            continue
        r = _run_model("ensemble", text[:MAX_TEXT_LEN])
        results.append({
            "text":       text[:80] + ("…" if len(text) > 80 else ""),
            "label":      r["label"],
            "confidence": r["confidence"],
            "score":      r["score"],
        })

    spam_count = sum(1 for r in results if r["label"] == "spam")
    return {
        "results":    results,
        "total":      len(results),
        "spam_count": spam_count,
        "ham_count":  len(results) - spam_count,
    }


@app.get("/models")
async def get_models():
    return {
        key: {
            "name":      info["name"],
            "accuracy":  info.get("accuracy"),
            "f1":        info.get("f1"),
            "precision": info.get("precision"),
            "recall":    info.get("recall"),
        }
        for key, info in _LOADED.items()
    }


@app.get("/health")
async def health():
    return {
        "status":        "ok",
        "models_loaded": list(_LOADED.keys()),
    }
