"""
train.py
--------
Run once to train and persist all three models.
Usage:
    python train.py
Outputs:
    models/naive_bayes.pkl
    models/logistic_regression.pkl
    models/ensemble.pkl
    models/metrics.json
"""

import os, json
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, accuracy_score,
    f1_score, precision_score, recall_score,
)
import joblib

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
DATA_PATH = os.environ.get("SPAM_DATA", "spam.tsv")

df = pd.read_csv(DATA_PATH, sep="\t", encoding="utf-8")
df.columns = ["label", "text"]
df = df[df["text"].notna() & df["label"].isin(["spam", "ham"])]
df["label"] = df["label"].str.strip().str.lower()

print(f"Dataset: {len(df)} samples  |  spam={df['label'].eq('spam').sum()}  ham={df['label'].eq('ham').sum()}")

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------------------------
# 2. Define pipelines
# ---------------------------------------------------------------------------
TFIDF_KWARGS = dict(
    max_features=8000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    strip_accents="unicode",
    analyzer="word",
    min_df=1,
)

pipelines = {
    "naive_bayes": Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_KWARGS)),
        ("clf",   MultinomialNB(alpha=0.1)),
    ]),
    "logistic_regression": Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_KWARGS)),
        ("clf",   LogisticRegression(C=5.0, max_iter=1000, solver="lbfgs")),
    ]),
}

# ---------------------------------------------------------------------------
# 3. Train, evaluate, save
# ---------------------------------------------------------------------------
os.makedirs("models", exist_ok=True)
metrics = {}

for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, pos_label="spam")
    prec = precision_score(y_test, y_pred, pos_label="spam")
    rec  = recall_score(y_test, y_pred, pos_label="spam")
    cv   = cross_val_score(pipe, X, y, cv=5, scoring="f1_macro").mean()

    metrics[name] = {
        "accuracy":  round(acc  * 100, 2),
        "f1":        round(f1,          4),
        "precision": round(prec,        4),
        "recall":    round(rec,         4),
        "cv_f1_macro": round(cv,        4),
    }

    joblib.dump(pipe, f"models/{name}.pkl")

    print(f"\n── {name} ──")
    print(classification_report(y_test, y_pred))

# ---------------------------------------------------------------------------
# 4. Soft-voting ensemble (uses predict_proba from both)
# ---------------------------------------------------------------------------
nb_pipe = joblib.load("models/naive_bayes.pkl")
lr_pipe = joblib.load("models/logistic_regression.pkl")

# VotingClassifier needs estimators that expose predict_proba
# We wrap the already-fitted pipelines directly
import numpy as np

# Manual soft-voting ensemble — avoid VotingClassifier wrapper complexity
nb_proba = nb_pipe.predict_proba(X_test)
lr_proba = lr_pipe.predict_proba(X_test)
classes  = nb_pipe.classes_   # ['ham', 'spam']

ens_proba    = 0.35 * nb_proba + 0.65 * lr_proba
y_pred_ens   = classes[ens_proba.argmax(axis=1)]

# Save ensemble as a simple dict of fitted pipelines + weights
ens = {"nb": nb_pipe, "lr": lr_pipe, "weights": [0.35, 0.65], "classes": list(classes)}

acc  = accuracy_score(y_test, y_pred_ens)
f1   = f1_score(y_test, y_pred_ens, pos_label="spam")
prec = precision_score(y_test, y_pred_ens, pos_label="spam")
rec  = recall_score(y_test, y_pred_ens, pos_label="spam")

metrics["ensemble"] = {
    "accuracy":  round(acc  * 100, 2),
    "f1":        round(f1,          4),
    "precision": round(prec,        4),
    "recall":    round(rec,         4),
    "cv_f1_macro": None,
}

joblib.dump(ens, "models/ensemble.pkl")
print(f"\n── ensemble ──")
print(classification_report(y_test, y_pred_ens))

# ---------------------------------------------------------------------------
# 5. Persist metrics so main.py can load them at startup
# ---------------------------------------------------------------------------
with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\n✓ All models saved to models/")
print(json.dumps(metrics, indent=2))
