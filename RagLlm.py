from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import (Settings, Document)
import faiss
import numpy as np
import ollama
from pydantic import BaseModel, ValidationError
from typing import List
import os

# Optionally import PyPDF2 in case a PDF file is used
try:
    import PyPDF2
except ImportError:
    print("PyPDF2 is not installed. Run 'pip install PyPDF2' if you need to extract text from PDFs.")


class ResponseModel(BaseModel):
    description: str
    similar: List[str]


def chat_with_llm(message_list, output_format):
    response = ollama.chat(model='deepseek-r1:8b', messages=message_list, format=output_format)
    return response.get("message", {}).get("content", "")


EMBED_MODEL = 'BAAI/bge-small-en-v1.5'
K = 1

Settings.embed_model = HuggingFaceEmbedding(
    model_name=EMBED_MODEL
)

docs = []
faiss_index = None

# Prompt for specific snake queries (e.g., "cobra")
PROMPT_SYS_INITIAL_RESPONSE_1 = """
    You're a helpful assistant who knows detailed information about venomous snakes in Sri Lanka.

    Here is some context related to the user question:
    -----------------------------------------
    {context_str}
    -----------------------------------------
    Considering the above information, answer the user question with details about the specific snake and suggest related species.

    Please keep in mind the following guidelines:
        - Provide detailed biological, ecological, and behavioral insights.
"""

# Prompt for general queries (e.g., "Who are the venomous snakes in Sri Lanka?")
PROMPT_SYS_INITIAL_RESPONSE_2 = """
    You're a helpful assistant who knows comprehensive information about venomous snakes in Sri Lanka.

    Here is some context related to the user question:
    -----------------------------------------
    {context_str}
    -----------------------------------------
    Considering the above information, answer the user question by listing the names and details of all five venomous snake species.

    Please keep in mind the following guidelines:
        - Provide an overview including key details and safety guidelines.
"""

# Define a list of known snake class names (in lowercase for matching)
known_classes = ["cobra", "commonkrait", "russellsviper", "sawscaledviper", "ceylonkrait"]


def init_():
    global docs, faiss_index

    # If data.txt exists, read from it; otherwise, try reading from a PDF
    if os.path.exists("data.txt"):
        with open("data.txt", 'r', encoding="UTF-8") as file:
            text = file.read()
    else:
        # Fallback to reading from a PDF file named "data.pdf"
        pdf_file_path = "data.pdf"  # Replace with your actual PDF file path if needed.
        text = ""
        if os.path.exists(pdf_file_path):
            with open(pdf_file_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        else:
            raise FileNotFoundError("Neither data.txt nor data.pdf was found.")

    # Create a Document with the extracted text
    docs.append(Document(text=text))

    # Generate embeddings for each document in docs
    embeddings = np.array([Settings.embed_model._embed(doc.text) for doc in docs])
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss_index = index

    print("Faiss index created")


def response_(user_question):
    # Compute embedding for the user question
    question_embedding = Settings.embed_model._embed(user_question)
    D, I = faiss_index.search(np.array([question_embedding]), k=K)
    top_docs = [docs[i] for i in I[0]]
    context_str = " ".join([doc.text for doc in top_docs])

    # Decide which prompt to use based on the user question
    if user_question.lower() in known_classes:
        system_prompt = PROMPT_SYS_INITIAL_RESPONSE_1.format(context_str=context_str)
    else:
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
            "similar": []
        }


# Initialize the FAISS index using the extracted text
init_()
