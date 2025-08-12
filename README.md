# 🧠 SafeSpace – AI Mental Health & Medical Assistant

SafeSpace is a powerful and supportive AI chatbot that provides empathetic mental health support and basic medical information. It can also analyze medical images (like X-rays, skin lesions, etc.) using OpenAI’s GPT-4o Vision model.

> ⚠️ **Disclaimer:** This tool does **not replace medical advice**. Always consult a licensed healthcare professional for medical concerns.
I use ollama for medgamma model. ollama run alibayram/medgemma

---

## 💡 Features

- 🤖 GPT-4 and alibayram/medgemma:4b-powered chat for mental & physical health queries
- 📍 Therapist recommendations (static list, customizable)
- 🆘 Emergency call integration via Twilio
- 📄 PDF medical encyclopedia search using FAISS
- 🖼️ Image understanding with GPT-4o Vision (X-rays, skin issues)
- 🗂️ Session-based chat history (Streamlit frontend)

---

## 🔧 Tech Stack

| Layer     | Tech                              |
|-----------|-----------------------------------|
| Backend   | FastAPI + LangGraph + OpenAI SDK  |
| Frontend  | Streamlit                         |
| NLP Model | `gpt-4` & `alibayram/medgemma:4b` |
| Image AI  | OpenAI GPT-4o Vision              |
| Search    | FAISS Vector DB + PDF embeddings  |
| Deploy    | Localhost or cloud-compatible     |

---


### 🧱 Prerequisites

- Python 3.10+
git clone https://github.com/ShubhamZoro/AI-Mental-Health-Chatbot.git

> Install `uv`: pip install uv
> uv sync


