"""Stage 9: LangChain agent that answers "is this flower open or closed?" using a local
Ollama LLM (text-only, no vision model) plus the trained classifier as a tool.

The flower name is only used by the LLM to phrase the final sentence - it is never fed
into the classifier, which decides purely from the image.
"""
from __future__ import annotations

import json

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from flower_state_ai import config
from flower_state_ai.inference import predict


class PredictFlowerStateInput(BaseModel):
    image_path: str = Field(description="Absolute or relative path to the flower photo on disk")


@tool("predict_flower_state", args_schema=PredictFlowerStateInput)
def predict_flower_state_tool(image_path: str) -> str:
    """Given a path to a photo of a flower, determine whether the flower in it is open or
    closed, with a confidence score. Always use this tool when asked about a flower's
    open/closed state."""
    try:
        result = predict(image_path)
        return json.dumps({"label": result.label, "confidence": round(result.confidence, 4)})
    except Exception as e:
        return json.dumps({"error": str(e)})


SYSTEM_PROMPT = (
    "You are a helpful assistant that tells users whether a flower in a photo is open or "
    "closed. Always call the predict_flower_state tool exactly once with the given image "
    "path before answering. Then phrase a single friendly sentence using the flower name "
    "the user gave you and the tool's label and confidence, e.g. 'The Rose in this photo "
    "looks open, about 93% confident.' Never invent a confidence value - use exactly what "
    "the tool returned. If the tool returns an error, tell the user plainly what went wrong."
)


def build_agent(model_name: str = config.OLLAMA_MODEL):
    llm = ChatOllama(model=model_name, temperature=0)
    return create_agent(llm, [predict_flower_state_tool], system_prompt=SYSTEM_PROMPT)


def ask(image_path: str, flower_name: str, executor=None) -> str:
    executor = executor or build_agent()
    human_input = f"Here is a photo at path '{image_path}' of a {flower_name}. Is it open or closed?"
    try:
        result = executor.invoke({"messages": [{"role": "user", "content": human_input}]})
    except Exception as e:
        return (
            f"Could not reach Ollama or run the agent ({e}). "
            f"Make sure 'ollama serve' is running and the model is pulled "
            f"(ollama pull {config.OLLAMA_MODEL})."
        )
    return result["messages"][-1].content
