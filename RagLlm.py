from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import (Settings, Document)
import faiss
import numpy as np
import ollama
from pydantic import BaseModel, ValidationError
from typing import List


class ResponseModel(BaseModel):
    description: str
    similar: List[str]


def chat_with_llm(message_list, output_format):
    response = ollama.chat(model='', messages=message_list, format=output_format)
    return response.get("message", {}).get("content", "")


EMBED_MODEL = 'BAAI/bge-small-en-v1.5'
K = 1

Settings.embed_model = HuggingFaceEmbedding(
    model_name=EMBED_MODEL
)

docs = []
faiss_index = None

PROMPT_SYS_INITIAL_RESPONSE_1 = """
    You're a helpful assistant who knows 

    Here is some context related to the user question:
    -----------------------------------------
    {context_str}
    -----------------------------------------
    Considering the above information and answer the user question.


    Please keep in mind the following guidelines:
        - Don't
"""

PROMPT_SYS_INITIAL_RESPONSE_2 = """
    You're a helpful assistant who knows 

    Here is some context related to the user question:
    -----------------------------------------
    {context_str}
    -----------------------------------------
    Considering the above information and answer the user question.


    Please keep in mind the following guidelines:
        - Don't
"""


def init_():
    global docs
    global faiss_index

    with open(".txt", 'r', encoding="UTF-8") as file:
        text = file.read()

    docs.append(Document(text=text))

    embeddings = np.array([Settings.embed_model._embed(doc.text) for doc in docs])
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss_index = index

    print("Faiss index created")


def response_(user_question):
    question_embedding = Settings.embed_model._embed(user_question)

    D, I = faiss_index.search(np.array([question_embedding]), k=K)
    top_docs = [docs[i] for i in I[0]]
    context_str = " ".join([doc.text for doc in top_docs])

    system_prompt = PROMPT_SYS_INITIAL_RESPONSE_2.format(context_str=context_str)

    message_list = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_question}
    ]

    try:
        response = chat_with_llm(message_list, ResponseModel.model_json_schema())
        parsed_response = ResponseModel.model_validate_json(response)
        print(parsed_response)
        return parsed_response
    except ValidationError as e:
        print("Validation failed:", e)
        return {
            "description": "Cannot respond",
            "similar": "Cannot respond"
        }