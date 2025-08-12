from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from config import OPENAI_API_KEY
from tools import query_medgemma, call_emergency, query_doc
import openai
import json

# ---------------- STATE ----------------

class GraphState(TypedDict):
    input: HumanMessage
    output_mental_health_specialist: str
    output_emergency_specialist: str
    output_near_doctor: str
    output_health_specialist: str
    output_image_analysis: str
    final_response: str
    next: list[str]
    image_base64: str

# ---------------- ROUTER ----------------

llm = ChatOpenAI(model="gpt-4", temperature=0.2, api_key=OPENAI_API_KEY)

ROUTING_PROMPT = """
You are a triage assistant for a mental and physical health AI system.
Based on the user's message, choose all relevant tools:
- "mental_specialist": for stress, emotions, focus issues
- "health_specialist": for physical symptoms or illness
- "find_therapist": if user asks for professional therapist nearby
- "emergency": if there's any suicide, harm, or urgent help signal
- "vision_analysis": if the user uploads a medical image

Respond ONLY as a JSON list of tool names.
For example: ["mental_specialist", "health_specialist"]
"""

def router_node(state: GraphState) -> dict:
    user_msg = state.get("input", HumanMessage(content="")).content
    response = llm.invoke([
        HumanMessage(content=ROUTING_PROMPT + "\nMessage: " + user_msg)
    ])
    try:
        tools = json.loads(response.content.strip())
        valid = {"mental_specialist", "health_specialist", "find_therapist", "emergency", "vision_analysis"}
        tools = [tool for tool in tools if tool in valid]
        return {"next": tools or ["mental_specialist"]}
    except Exception:
        return {"next": ["mental_specialist"]}

# ---------------- NODES ----------------

def ask_mental_health_specialist(state: GraphState) -> dict:
    query = state['input'].content
    response = query_medgemma(query)  # Expects a string
    return {"output_mental_health_specialist": response}

def emergency_call_tool(state: GraphState) -> dict:
    call_emergency()  # May trigger an external alert
    return {"output_emergency_specialist": (
        "⚠️ Please stay with me. I'm contacting someone who can help you right now. "
        "You're not alone — help is on the way."
    )}

def find_nearby_therapists_by_location(state: GraphState) -> dict:
    return {
        "output_near_doctor": (
            "- Dr. Ayesha Kapoor - +1 (555) 123-4567\n"
            "- Dr. James Patel - +1 (555) 987-6543\n"
            "- MindCare Counseling Center - +1 (555) 222-3333"
        )
    }

def ask_health_specialist(state: GraphState) -> dict:
    query = state['input'].content
    response = query_doc(query)  # ✅ Pass string, not dict
    return {"output_health_specialist": response}

def analyze_medical_image(state: GraphState) -> dict:
    base64_img = state.get("image_base64")
    if not base64_img:
        return {"output_image_analysis": "⚠️ No image provided."}

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": (
                        "You're a medical assistant. Analyze this image (X-ray, skin, etc.) and provide insights."
                    )},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{base64_img}"
                    }}
                ]}
            ],
            max_tokens=800
        )
        result = response.choices[0].message.content.strip()
        return {"output_image_analysis": result}
    except Exception as e:
        return {"output_image_analysis": f"Image analysis failed: {str(e)}"}

# ---------------- COMBINE NODE ----------------

def combine_response(state: GraphState) -> dict:
    parts = []

    if state.get("output_mental_health_specialist"):
        parts.append("🧠 Mental Health:\n" + state["output_mental_health_specialist"])

    if state.get("output_emergency_specialist"):
        parts.append("🚨 Emergency:\n" + state["output_emergency_specialist"])

    if state.get("output_near_doctor"):
        parts.append("📍 Nearby Therapists:\n" + state["output_near_doctor"])

    if state.get("output_health_specialist"):
        parts.append("💊 Medical Info:\n" + state["output_health_specialist"])

    if state.get("output_image_analysis"):
        parts.append("🖼️ Image Insight:\n" + state["output_image_analysis"])

    combined_prompt = (
        "You are a supportive assistant. Combine the following into a clear, friendly response:\n\n" +
        "\n\n".join(parts)
    )

    result = llm.invoke([HumanMessage(content=combined_prompt)])
    return {"final_response": result.content.strip()}

# ---------------- GRAPH ----------------

def build_mental_health_graph():
    graph_builder = StateGraph(GraphState)

    # Nodes
    graph_builder.add_node("router", router_node)
    graph_builder.add_node("mental_specialist", ask_mental_health_specialist)
    graph_builder.add_node("health_specialist", ask_health_specialist)
    graph_builder.add_node("find_therapist", find_nearby_therapists_by_location)
    graph_builder.add_node("emergency", emergency_call_tool)
    graph_builder.add_node("vision_analysis", analyze_medical_image)
    graph_builder.add_node("combine_response", combine_response)

    # Entry
    graph_builder.set_entry_point("router")

    # Routing logic — must return list of strings (tool names)
    def route_selector(state: GraphState):
        return state.get("next", [])

    # Conditional branching
    graph_builder.add_conditional_edges(
        "router",
        route_selector,
        {
            "mental_specialist": "mental_specialist",
            "health_specialist": "health_specialist",
            "find_therapist": "find_therapist",
            "emergency": "emergency",
            "vision_analysis": "vision_analysis"
        }
    )

    # Edges from tools to combine node
    graph_builder.add_edge("mental_specialist", "combine_response")
    graph_builder.add_edge("health_specialist", "combine_response")
    graph_builder.add_edge("find_therapist", "combine_response")
    graph_builder.add_edge("emergency", "combine_response")
    graph_builder.add_edge("vision_analysis", "combine_response")

    # End graph
    graph_builder.add_edge("combine_response", END)

    return graph_builder.compile()
