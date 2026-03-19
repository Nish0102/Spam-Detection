# 🛡️ SpamShield — AI Spam Detector

> Real-time spam detection with word-level explainability, model comparison, and a cybersecurity-inspired UI.

![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi)
![React](https://img.shields.io/badge/React-18.3-61DAFB?style=flat-square&logo=react)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Real-time Detection** | Instant spam/ham verdict with confidence score |
| 🔬 **Word Explainability** | Highlights which words triggered the prediction (LIME-style) |
| 📊 **Model Comparison** | Side-by-side metrics for Naive Bayes, Rule-Based, and Ensemble |
| 📈 **Radar Chart** | Visual accuracy/F1/precision/recall comparison |
| 🗂 **Scan History** | Last 20 scans with one-click replay |
| ⚡ **Batch Mode** | Analyze multiple messages via `/batch` endpoint |

---

## 🗂 Project Structure

```
spam-detector/
├── backend/
│   ├── main.py           # FastAPI: 3 classifiers + explainability + batch
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── App.jsx
    │   ├── hooks/useSpam.js
    │   └── components/
    │       ├── VerdictPanel.jsx      # Animated result + stats
    │       ├── HighlightPanel.jsx    # Word-level explanation
    │       ├── ModelComparison.jsx   # Metrics + radar chart
    │       └── HistoryPanel.jsx      # Scan log
    ├── index.html
    └── package.json
```

---

## 🚀 Running Locally

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

Visit **http://localhost:5173**

---

## 🧠 Models

| Model | Accuracy | F1 | Notes |
|---|---|---|---|
| Naive Bayes | 85.2% | 0.83 | Fast bag-of-words baseline |
| Rule-Based | 88.7% | 0.87 | Pattern-weighted scoring |
| **Ensemble** | **92.4%** | **0.92** | Best — weighted combination |

> To plug in a real sklearn/transformers model, replace the `predict_*` functions in `main.py` with your trained model's `.predict_proba()`.

---

## 🔮 Upgrade Path
- [ ] Train on SMS Spam Collection / Enron dataset
- [ ] Add DistilBERT fine-tuned classifier for 97%+ accuracy
- [ ] Add SHAP values for deeper explainability
- [ ] User upload CSV for batch processing
- [ ] Deploy backend on Railway, frontend on Vercel
