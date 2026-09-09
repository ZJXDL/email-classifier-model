# 📧 Smart Email Classifier — ML Engineering Portfolio Project

A lightweight, production-grade email spam detector built with Python, scikit-learn, and deployed live via Streamlit.

## 🚀 Features
- ✅ Trained on 5,574 real SMS/email messages (spam/ham labels)
- ✅ Uses TF-IDF + Naive Bayes — fast, accurate, and interpretable
- ✅ Live web interface: paste any text, get instant classification
- ✅ Deployed publicly: [https://your-app-name.streamlit.app](https://your-app-name.streamlit.app)
- ✅ Logs predictions internally for future improvement

## 🛠️ Tech Stack
- Python 3.11+
- scikit-learn (TF-IDF, Naive Bayes)
- Streamlit (for web UI)
- joblib (model serialization)
- GitHub + Streamlit Cloud (deployment)

## 💡 Why This Matters
This isn’t just a toy project — it’s a complete ML pipeline:
> **Data → Training → Serving → Deployment → Logging**

Exactly what entry-level ML engineers do at startups and tech companies.  
I built this in under 2 days — no GPU, no cloud credits, no complex frameworks.

## 📂 How to Run Locally
1. Clone this repo  
2. Install dependencies:  
   ```bash
   pip install scikit-learn pandas joblib streamlit
