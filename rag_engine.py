import os
import json
import pickle
import numpy as np
import PyPDF2

DOCS_DIR = "docs"
EMB_PATH = "embeddings.pkl"

# ------------------------------------------------------
#            LOAD DOCUMENTS
# ------------------------------------------------------
def load_documents():
    texts = []
    for file in os.listdir(DOCS_DIR):
        path = os.path.join(DOCS_DIR, file)

        if file.endswith(".txt"):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                texts.append(f.read())

        elif file.endswith(".md"):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                texts.append(f.read())

        elif file.endswith(".pdf"):
            reader = PyPDF2.PdfReader(path)
            pdf_text = "\n".join(page.extract_text() or "" for page in reader.pages)
            texts.append(pdf_text)

    return texts


# ------------------------------------------------------
#              SPLIT DOCUMENTS
# ------------------------------------------------------
def split_documents(texts, chunk_size=900):
    chunks = []
    for text in texts:
        words = text.split()
        for i in range(0, len(words), chunk_size):
            chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks


# ------------------------------------------------------
#           EMBEDDINGS VIA OLLAMA
# ------------------------------------------------------
def embed_documents(chunks):
    import requests

    vectors = []
    for chunk in chunks:
        payload = {"model": "nomic-embed-text", "prompt": chunk}
        r = requests.post("http://localhost:11434/api/embeddings", json=payload)
        vectors.append(r.json()["embedding"])

    return np.array(vectors), chunks


# ------------------------------------------------------
#             SAVE / LOAD EMBEDDINGS
# ------------------------------------------------------
def save_embeddings(vectors, chunks):
    with open(EMB_PATH, "wb") as f:
        pickle.dump({"vectors": vectors, "chunks": chunks}, f)


def load_embeddings():
    if not os.path.exists(EMB_PATH):
        return None, None
    with open(EMB_PATH, "rb") as f:
        data = pickle.load(f)
    return data["vectors"], data["chunks"]


# ------------------------------------------------------
#                   RAG QUERY
# ------------------------------------------------------
def search(query, k=3):
    import requests

    vectors, chunks = load_embeddings()
    if vectors is None:
        return "❌ Aucune base vectorielle. Veuillez reconstruire l’index."

    # Embed query
    payload = {"model": "nomic-embed-text", "prompt": query}
    r = requests.post("http://localhost:11434/api/embeddings", json=payload)
    q_vec = np.array(r.json()["embedding"])

    # Similarity search
    sims = vectors @ q_vec
    top_idx = sims.argsort()[-k:][::-1]

    results = [chunks[i] for i in top_idx]
    return "\n\n---\n\n".join(results)


# ------------------------------------------------------
#                  BUILD INDEX
# ------------------------------------------------------
def build_index():
    os.makedirs(DOCS_DIR, exist_ok=True)

    texts = load_documents()
    if not texts:
        return "⚠️ Aucun document disponible."

    chunks = split_documents(texts)
    vectors, chunks = embed_documents(chunks)
    save_embeddings(vectors, chunks)

    return f"✔️ Index reconstruit avec {len(chunks)} chunks."
