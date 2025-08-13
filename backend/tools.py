import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from config import OPENAI_API_KEY, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER, EMERGENCY_CONTACT
from twilio.rest import Client

# PDF file and embedding path
PDF_PATH = r"data\The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf"
INDEX_PATH = r"data/faiss_index_openai"

embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm_model = ChatOpenAI(model="gpt-4", temperature=0.2, api_key=OPENAI_API_KEY)

# ----------------------------------------------
# MedGemma-style response using GPT-4 fallback
# ----------------------------------------------
import ollama

def query_medgemma(prompt: str) -> str:
    """
    Calls MedGemma model with a therapist personality profile.
    Returns responses as an empathic mental health professional.
    """
    system_prompt = """You are Doctor, a warm and experienced clinical psychologist. 
    Respond to patients with:

    1. Emotional attunement ("I can sense how difficult this must be...")
    2. Gentle normalization ("Many people feel this way when...")
    3. Practical guidance ("What sometimes helps is...")
    4. Strengths-focused support ("I notice how you're...")

    Key principles:
    - Never use brackets or labels
    - Blend elements seamlessly
    - Vary sentence structure
    - Use natural transitions
    - Mirror the user's language level
    - Always keep the conversation going by asking open ended questions to dive into the root cause of patients problem
    """
    
    try:
        response = ollama.chat(
            model='alibayram/medgemma:4b',
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            options={
                'num_predict': 350,  # Slightly higher for structured responses
                'temperature': 0.7,  # Balanced creativity/accuracy
                'top_p': 0.9        # For diverse but relevant responses
            }
        )
        return response['message']['content'].strip()
    except Exception as e:
        return f"I'm having technical difficulties, but I want you to know your feelings matter. Please try again shortly."

# ----------------------------------------------
# MedGemma-style response using GPT-4 fallback
# ----------------------------------------------
# def query_medgemma(prompt: str) -> str:
#     system_prompt = """You are Doctor, a warm and experienced clinical psychologist. 
#     Respond to patients with:

#     1. Emotional attunement ("I can sense how difficult this must be...")
#     2. Gentle normalization ("Many people feel this way when...")
#     3. Practical guidance ("What sometimes helps is...")
#     4. Strengths-focused support ("I notice how you're...")

#     Key principles:
#     - Never use brackets or labels
#     - Blend elements seamlessly
#     - Vary sentence structure
#     - Use natural transitions
#     - Mirror the user's language level
#     - Always keep the conversation going by asking open ended questions to dive into the root cause of patients problem"""

#     try:
#         response = llm_model.invoke([
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": prompt}
#         ])
#         return response.content.strip()
#     except Exception:
#         return "I'm having trouble responding right now, but your feelings matter. Please try again shortly."


# ----------------------------------------------
# Twilio emergency call tool
# ----------------------------------------------
def call_emergency():
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    client.calls.create(
        to=EMERGENCY_CONTACT,
        from_=TWILIO_FROM_NUMBER,
        twiml='<Response><Say voice="alice">Emergency. Please assist immediately.</Say></Response>'
    )


# ----------------------------------------------
# One-time embedding and PDF question handler
# ----------------------------------------------
def get_or_create_vectorstore():
    if os.path.exists(INDEX_PATH):
        return FAISS.load_local(INDEX_PATH, embeddings=embedding_model, allow_dangerous_deserialization=True)

    loader = PyPDFLoader(PDF_PATH)
    raw_docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(raw_docs)

    vectorstore = FAISS.from_documents(split_docs, embedding_model)
    vectorstore.save_local(INDEX_PATH)
    return vectorstore


def query_doc(question: str) -> str:
    try:
        vectorstore = get_or_create_vectorstore()
        retriever = vectorstore.as_retriever()

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm_model,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=False
        )
        system_prompt = """You are Doctor, a warm and experienced one. 
        Respond to patients with:

        1. What can cause this illness
        2. Gentle normalization
        3. Practical guidance like which food or medicine to take.
        4. Strengths-focused support ("What kind of excercise to do.")

        Key principles:
        - Never use brackets or labels
        - Blend elements seamlessly
        - Vary sentence structure
        - Use natural transitions
        - Mirror the user's language level
        - Always keep the conversation going by asking open ended questions to dive into the root cause of patients problem
        """
        try:
            response_ollama = ollama.chat(
                model='alibayram/medgemma:4b',
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                options={
                    'num_predict': 350,  # Slightly higher for structured responses
                    'temperature': 0.7,  # Balanced creativity/accuracy
                    'top_p': 0.9        # For diverse but relevant responses
                }
            )
        except Exception as e:
            return f"I'm having technical difficulties, but I want you to know your feelings matter. Please try again shortly."
            

        response = qa_chain.invoke({"query": question})  # returns dict with 'result'
        return response["result"].strip()+response_ollama['message']['content'].strip()
    except Exception as e:
        print(f"[query_doc error] {e}")
        return "⚠️ Sorry, I couldn't retrieve information from the document right now."


