from fastapi import FastAPI, Query
import requests
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# === FastAPI App ===
app = FastAPI()

# === Config ===
API_URL = "https://november7-730026606190.europe-west1.run.app/messages"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DIMENSION = 384
TOP_K = 3  # default top similar vectors

# === Load embedding model ===
embedder = SentenceTransformer(EMBEDDING_MODEL)

# === Initialize FAISS index ===
index = faiss.IndexFlatL2(DIMENSION)
stored_texts = []

# === Load CPU-friendly open-source LLM from transformers ===
LLM_MODEL = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)
llm = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)

# === Endpoints ===

@app.get("/")
def root():
    return {"message": "RAG QA System with Transformers LLM Running"}


@app.get("/fetch_and_store")
def fetch_and_store():
    response = requests.get(API_URL)
    if response.status_code != 200:
        return {"status": "error", "code": response.status_code}

    data = response.json()
    if "items" not in data or not data["items"]:
        return {"status": "error", "message": "No items found in API response."}

    texts = [item.get("message") for item in data["items"] if "message" in item]
    if not texts:
        return {"status": "error", "message": "No messages found in API items."}

    embeddings = embedder.encode(texts, convert_to_numpy=True)
    index.add(embeddings)
    stored_texts.extend(texts)

    return {
        "status": "success",
        "num_vectors_added": len(texts),
        "total_vectors_in_faiss": index.ntotal
    }


@app.get("/ask")
def search_and_answer(
    query: str = Query(..., description="User query to answer")
    # top_k: int = Query(TOP_K, description="Number of top similar vectors to use as context")
):
    if index.ntotal == 0:
        # Call the fetch_and_store function
        fetch_response = fetch_and_store()
        if fetch_response.get("status") != "success":
            return {"status": "error", "message": "Failed to fetch data before search."}
        

    # Step 1: Embed the query
    query_vector = embedder.encode([query], convert_to_numpy=True)

    # Step 2: Search FAISS
    distances, indices = index.search(query_vector, k=100)

    # Step 3: Prepare context from top vectors
    context_texts = [stored_texts[idx] for idx in indices[0] if idx < len(stored_texts)]
    context_str = "\n".join(context_texts)

    # Step 4: Create prompt for LLM
    prompt = f"Answer the question in 1-2 sentences based on the context below.\n\nContext:\n{context_str}\n\nQuestion: {query}\nAnswer:"

    # Step 5: Generate answer using LLM
    result = llm(prompt)
    answer = result[0]['generated_text']

    return {
        "query": query,
        # "context_used": context_texts,
        "llm_answer": answer
    }
