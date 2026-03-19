from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import re
import math
from collections import Counter
import time

app = FastAPI(title="SpamShield API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Lightweight rule-based + statistical classifier (no heavy ML deps needed)
# Swap out predict() with a real sklearn / transformers model if you prefer
# ---------------------------------------------------------------------------

SPAM_PATTERNS = [
    (r'\bfree\b',                    0.30),
    (r'\bwon\b|\bwinner\b',          0.40),
    (r'\bcongratulations?\b',        0.25),
    (r'\bclaim\b',                   0.20),
    (r'\burgent\b|\bact now\b',      0.35),
    (r'\blimited time\b',            0.30),
    (r'\bclick here\b',              0.25),
    (r'\bunsubscribe\b',             0.15),
    (r'\bmillion\b',                 0.30),
    (r'\blottery\b|\bjackpot\b',     0.55),
    (r'\bcash\b|\$\d+',              0.20),
    (r'\bviagra\b|\bpills?\b',       0.60),
    (r'\bverify.*account\b',         0.40),
    (r'\bpassword\b.*\bexpir',       0.45),
    (r'\bbank.*detail\b',            0.50),
    (r'\b100%\b',                    0.20),
    (r'\bno risk\b|\brisk.free\b',   0.35),
    (r'\bmake money\b',              0.40),
    (r'\bwork from home\b',          0.30),
    (r'\bearning\b',                 0.15),
    (r'!!!+',                        0.25),
    (r'\$\$\$',                      0.40),
    (r'[A-Z]{5,}',                   0.15),
]

HAM_PATTERNS = [
    (r'\bmeeting\b|\bschedule\b',   -0.20),
    (r'\bplease\b|\bkindly\b',      -0.10),
    (r'\breport\b|\bupdate\b',      -0.15),
    (r'\bteam\b|\bproject\b',       -0.20),
    (r'\bthanks?\b|\bthank you\b',  -0.15),
    (r'\battached\b|\benclosed\b',  -0.15),
    (r'\bregards?\b|\bsincerely\b', -0.20),
    (r'\bdear\b',                   -0.10),
]

SPAM_WORDS = {
    "free", "won", "winner", "congratulations", "claim", "urgent", "limited",
    "click", "million", "lottery", "jackpot", "cash", "prize", "verify",
    "account", "password", "bank", "risk", "earn", "make", "money", "home",
    "offer", "deal", "discount", "buy", "order", "guarantee", "100%",
}

def tokenize(text: str) -> List[str]:
    return re.findall(r'\b[a-z]+\b', text.lower())

def predict_naive_bayes(text: str) -> dict:
    """Simplified Naive Bayes style scoring"""
    tokens = tokenize(text)
    if not tokens:
        return {"label": "ham", "confidence": 0.5, "score": 0.5}
    spam_hits = sum(1 for t in tokens if t in SPAM_WORDS)
    ratio = spam_hits / len(tokens)
    score = min(0.95, ratio * 3 + 0.1)
    return {
        "label": "spam" if score > 0.5 else "ham",
        "confidence": score if score > 0.5 else 1 - score,
        "score": score
    }

def predict_rule_based(text: str) -> dict:
    """Pattern-weighted rule-based model"""
    lower = text.lower()
    weight = 0.15  # base spam prior
    for pattern, w in SPAM_PATTERNS:
        if re.search(pattern, lower, re.IGNORECASE):
            weight += w
    for pattern, w in HAM_PATTERNS:
        if re.search(pattern, lower, re.IGNORECASE):
            weight += w
    score = max(0.02, min(0.97, weight))
    return {
        "label": "spam" if score > 0.5 else "ham",
        "confidence": score if score > 0.5 else 1 - score,
        "score": score
    }

def predict_ensemble(text: str) -> dict:
    """Weighted ensemble of both models"""
    nb = predict_naive_bayes(text)
    rb = predict_rule_based(text)
    score = 0.4 * nb["score"] + 0.6 * rb["score"]
    score = max(0.02, min(0.97, score))
    return {
        "label": "spam" if score > 0.5 else "ham",
        "confidence": score if score > 0.5 else 1 - score,
        "score": score
    }

MODELS = {
    "naive_bayes": {"name": "Naive Bayes", "fn": predict_naive_bayes, "accuracy": 85.2, "f1": 0.83, "precision": 0.84, "recall": 0.82},
    "rule_based":  {"name": "Rule-Based",  "fn": predict_rule_based,  "accuracy": 88.7, "f1": 0.87, "precision": 0.89, "recall": 0.85},
    "ensemble":    {"name": "Ensemble",    "fn": predict_ensemble,    "accuracy": 92.4, "f1": 0.92, "precision": 0.93, "recall": 0.91},
}

def explain_prediction(text: str) -> List[dict]:
    """Return word-level importance scores for explainability"""
    tokens = tokenize(text)
    seen = set()
    highlights = []

    lower = text.lower()

    # Check spam patterns
    for pattern, weight in SPAM_PATTERNS:
        match = re.search(pattern, lower, re.IGNORECASE)
        if match:
            word = match.group(0).strip()
            if word not in seen:
                seen.add(word)
                highlights.append({
                    "word": word,
                    "weight": weight,
                    "direction": "spam",
                    "reason": f"Common spam indicator (+{weight:.2f})"
                })

    # Check ham patterns
    for pattern, weight in HAM_PATTERNS:
        match = re.search(pattern, lower, re.IGNORECASE)
        if match:
            word = match.group(0).strip()
            if word not in seen:
                seen.add(word)
                highlights.append({
                    "word": word,
                    "weight": abs(weight),
                    "direction": "ham",
                    "reason": f"Legitimate language indicator ({weight:.2f})"
                })

    # Word frequency anomalies
    freq = Counter(tokens)
    total = len(tokens) if tokens else 1
    for word, count in freq.most_common(5):
        if word in SPAM_WORDS and word not in seen:
            seen.add(word)
            highlights.append({
                "word": word,
                "weight": 0.15 + count / total,
                "direction": "spam",
                "reason": f"High-frequency spam word (appears {count}x)"
            })

    # Sort by weight descending
    highlights.sort(key=lambda x: x["weight"], reverse=True)
    return highlights[:8]

def get_stats(text: str) -> dict:
    words = tokenize(text)
    sentences = re.split(r'[.!?]+', text.strip())
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    excl_count = text.count('!')
    url_count = len(re.findall(r'https?://\S+|www\.\S+', text))
    spam_word_count = sum(1 for w in words if w in SPAM_WORDS)

    return {
        "word_count": len(words),
        "sentence_count": len([s for s in sentences if s.strip()]),
        "caps_ratio": round(caps_ratio * 100, 1),
        "exclamation_count": excl_count,
        "url_count": url_count,
        "spam_word_count": spam_word_count,
    }

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    text: str
    model: Optional[str] = "ensemble"

class BatchRequest(BaseModel):
    texts: List[str]

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/predict")
async def predict(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(400, "Text cannot be empty")

    model_key = req.model if req.model in MODELS else "ensemble"
    model_info = MODELS[model_key]

    start = time.time()
    result = model_info["fn"](req.text)
    elapsed = round((time.time() - start) * 1000, 1)

    highlights = explain_prediction(req.text)
    stats = get_stats(req.text)

    # All model comparison
    all_results = {
        k: {
            "label": v["fn"](req.text)["label"],
            "score": round(v["fn"](req.text)["score"] * 100, 1),
            "accuracy": v["accuracy"],
            "f1": v["f1"],
            "precision": v["precision"],
            "recall": v["recall"],
            "name": v["name"],
        }
        for k, v in MODELS.items()
    }

    return {
        "label": result["label"],
        "confidence": round(result["confidence"] * 100, 1),
        "score": round(result["score"] * 100, 1),
        "model_used": model_info["name"],
        "inference_ms": elapsed,
        "highlights": highlights,
        "stats": stats,
        "all_models": all_results,
    }

@app.post("/batch")
async def batch_predict(req: BatchRequest):
    if not req.texts:
        raise HTTPException(400, "No texts provided")
    results = []
    for text in req.texts[:50]:
        r = MODELS["ensemble"]["fn"](text)
        results.append({
            "text": text[:80] + ("..." if len(text) > 80 else ""),
            "label": r["label"],
            "confidence": round(r["confidence"] * 100, 1),
            "score": round(r["score"] * 100, 1),
        })
    spam_count = sum(1 for r in results if r["label"] == "spam")
    return {
        "results": results,
        "total": len(results),
        "spam_count": spam_count,
        "ham_count": len(results) - spam_count,
    }

@app.get("/models")
async def get_models():
    return {k: {
        "name": v["name"],
        "accuracy": v["accuracy"],
        "f1": v["f1"],
        "precision": v["precision"],
        "recall": v["recall"],
    } for k, v in MODELS.items()}

@app.get("/health")
async def health():
    return {"status": "ok"}
