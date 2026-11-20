import os
import numpy as np
import faiss

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 1. LOAD PDF
def load_pdf(pdf_path):

    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    return text


# 2. CREATE CHUNKS
def create_chunks(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(text)

    return chunks


# 3. BUILD VECTORS
def build_vectors(chunks):

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=chunks
    )

    embeddings = [
        item.embedding for item in response.data
    ]

    vectors = np.array(
        embeddings,
        dtype="float32"
    )

    faiss.normalize_L2(vectors)

    return vectors


# 4. LOAD DATABASE
def load_database(vectors):

    dimension = vectors.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(vectors)

    return index


# 5. RETRIEVE INFORMATION
def retrieve(question, index, chunks):

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[question]
    )

    query_vector = np.array(
        [response.data[0].embedding],
        dtype="float32"
    )

    faiss.normalize_L2(query_vector)

    scores, indices = index.search(
        query_vector, 5
    )

    results = []

    for i in indices[0]:

        if i >= 0:
            results.append(chunks[i])

    return results


# 6. ASK RAG
def ask_rag(question, index, chunks):

    results = retrieve(
        question, index, chunks
    )

    context = "\n\n".join(results)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """
                You are a Karnataka Tourism AI Assistant.
                Answer using only the given context.
                Use simple English.
                Do not invent information.
                """
            },
            {
                "role": "user",
                "content": f"""
                Question: {question}

                Context: {context}

                Give a clear answer.
                """
            }
        ]
    )

    return response.choices[0].message.content
