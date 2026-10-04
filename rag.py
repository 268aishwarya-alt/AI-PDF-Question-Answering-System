import requests
import streamlit as st

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:1b"


def ask_llm(question, context, max_tokens=150):

    prompt = f"""
You are an academic assistant.

Use ONLY the academic context provided below.

CONTEXT:
{context}

QUESTION:
{question}

Give a short, clear and useful answer.
Do not invent information.

ANSWER:
"""

    response = requests.post(
        OLLAMA_URL,
        json={
    "model": MODEL_NAME,
    "prompt": prompt,
    "stream": True,
    "keep_alive": "10m",
    "options": {
        "temperature": 0.1,
        "num_predict": max_tokens,
        "num_ctx": 2048
    }
},
        stream=True,
        timeout=120
    )

    response.raise_for_status()

    answer = ""

    placeholder = st.empty()

    for line in response.iter_lines():

        if line:

            import json

            data = json.loads(line)

            if "response" in data:

                answer += data["response"]

                placeholder.markdown(
                    answer
                )

            if data.get("done", False):
                break

    return ""