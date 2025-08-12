# 🧠 SafeSpace – AI Mental Health & Medical Assistant

SafeSpace is a powerful and supportive AI chatbot that provides empathetic mental health support and basic medical information. It can also analyze medical images (like X-rays, skin lesions, etc.) using OpenAI’s GPT-4o Vision model.

> ⚠️ **Disclaimer:** This tool does **not replace medical advice**. Always consult a licensed healthcare professional for medical concerns.

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

## 📂 Project Structure

├── backend/
│ ├── main.py # FastAPI backend
│ ├── ai_agent.py # LangGraph logic
│ ├── tools.py # Tool handlers (PDF search, emergency)
│ ├── config.py # Env vars (keys, Twilio)
│ └── data/
│ └── The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf
│ └── faiss_index_openai/
├── frontend.py # Streamlit chat interface
├── requirements.txt
└── README.md

### 🧱 Prerequisites

- Python 3.10+
git clone https://github.com/ShubhamZoro/AI-Mental-Health-Chatbot.git

> Install `uv`: pip install uv
> uv sync

