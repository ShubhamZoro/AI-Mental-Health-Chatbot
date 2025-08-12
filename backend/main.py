from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from ai_agent import build_mental_health_graph
from config import OPENAI_API_KEY
import openai
from io import BytesIO
from PIL import Image, UnidentifiedImageError
import base64
import uvicorn
from openai import OpenAI
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

# ---------- Init ----------
app = FastAPI()
openai.api_key = OPENAI_API_KEY
graph = build_mental_health_graph()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Text Query Endpoint ----------
class Query(BaseModel):
    message: str

@app.post("/ask")
async def ask(query: Query):
    user_input = HumanMessage(content=query.message)
    result = graph.invoke({"input": user_input})
    return {"response": result["final_response"]}


# ---------- Utility ----------
def encode_image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


# ---------- Vision Upload Endpoint ----------


@app.post("/upload-image-openai")
async def upload_image_openai(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        # Validate image
        try:
            image = Image.open(BytesIO(contents)).convert("RGB")
        except UnidentifiedImageError:
            return {"error": "Uploaded file is not a valid image."}

        # Encode as base64 for GPT-4 Vision
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        base64_img = base64.b64encode(buffer.getvalue()).decode("utf-8")
        text_prompt = (
            "You're a helpful assistant. Please analyze the uploaded medical image "
            "and describe what patterns, textures, or visible details are noticeable. "
            "Do not diagnose. Just describe relevant features you observe in the image and tell is there any reason to go and see doctor."
        )
        # GPT-4 Vision prompt
        response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{base64_img}"
                    }}
                ]
            }
        ],
        max_tokens=800
    )


        result = response.choices[0].message.content
        return {"diagnosis": result}

    except Exception as e:
        return {"error": f"OpenAI Vision failed: {e}"}



# ---------- Start ----------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
