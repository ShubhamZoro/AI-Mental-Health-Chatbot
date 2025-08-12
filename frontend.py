import streamlit as st
import requests
from PIL import Image

st.set_page_config(page_title="🧠 SafeSpace Assistant", layout="wide")
st.title("🧠 SafeSpace – AI Mental Health & Medical Assistant")

# -------------------------------
# Initialize session state
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------
# Display previous messages
# -------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# -------------------------------
# Chat input for text
# -------------------------------
text_input = st.chat_input("What's on your mind?")

if text_input:
    st.session_state.messages.append({"role": "user", "content": text_input})
    with st.chat_message("user"):
        st.write(text_input)

    try:
        response = requests.post("http://localhost:8000/ask", json={"message": text_input})
        if response.status_code == 200:
            data = response.json()
            assistant_response = data.get("response", "⚠️ No response provided.")
        else:
            assistant_response = f"⚠️ Backend error: Status code {response.status_code}"
    except requests.exceptions.JSONDecodeError:
        assistant_response = "⚠️ Backend returned invalid JSON."
    except Exception as e:
        assistant_response = f"⚠️ Request failed: {e}"

    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    with st.chat_message("assistant"):
        st.write(assistant_response)

# -------------------------------
# Upload and process medical image
# -------------------------------
st.divider()
st.subheader("📷 Upload an image for medical insight")

image_file = st.file_uploader("Upload an image (e.g. skin lesion, X-ray)", type=["png", "jpg", "jpeg"])

if image_file:
    try:
        image = Image.open(image_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
    except Exception as e:
        st.error(f"❌ Error displaying image: {e}")

    with st.spinner("Analyzing image with GPT-4 Vision..."):
        try:
            image_file.seek(0)
            file_bytes = image_file.read()

            files = {
                "file": (image_file.name, file_bytes, image_file.type)
            }
            response = requests.post("http://localhost:8000/upload-image-openai", files=files)
            try:
                result = response.json()
            except requests.exceptions.JSONDecodeError:
                result = {"error": "⚠️ Backend did not return valid JSON."}
        except Exception as e:
            result = {"error": f"Image analysis failed: {e}"}

    st.subheader("🩺 GPT-4 Vision Diagnosis")
    if "diagnosis" in result:
        st.success(result["diagnosis"])
        st.session_state.messages.append({
            "role": "assistant",
            "content": "🖼️ Image Insight:\n" + result["diagnosis"]
        })
    else:
        st.error(result.get("error", "Something went wrong."))


