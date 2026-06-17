import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import gradio as gr

# ── 1. CONFIG ──────────────────────────────────────────────
DOCS_FOLDER = "documents"
EMBED_MODEL  = "all-MiniLM-L6-v2"
TOP_K        = 3

# ── 2. LOAD LOCAL LLM (no API key needed) ─────────────────
print("Loading LLM model (first time may take 1-2 min to download)...")
generator = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
print("LLM ready!")

# ── 3. LOAD DOCUMENTS ──────────────────────────────────────
def load_documents(folder):
    chunks = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if filename.endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            words = text.split()
            for i in range(0, len(words), 200):
                chunk = " ".join(words[i:i+200])
                chunks.append({"text": chunk, "source": filename})
    return chunks

# ── 4. BUILD VECTOR INDEX ──────────────────────────────────
def build_index(chunks, model):
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

# ── 5. RETRIEVE RELEVANT CHUNKS ────────────────────────────
def retrieve(query, model, index, chunks, top_k=TOP_K):
    query_embedding = model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

# ── 6. GENERATE ANSWER ─────────────────────────────────────
def generate_answer(query, retrieved_chunks):
    context = " ".join([c["text"] for c in retrieved_chunks])[:800]
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    result = generator(prompt, max_new_tokens=150, do_sample=False, temperature=1.0)
    full_text = result[0]["generated_text"]
    # Extract only the answer part after "Answer:"
    answer = full_text.split("Answer:")[-1].strip()
    return answer

# ── 7. SETUP ───────────────────────────────────────────────
embed_model = SentenceTransformer(EMBED_MODEL)
print("Loading documents...")
chunks = load_documents(DOCS_FOLDER)
print(f"Loaded {len(chunks)} chunks.")
print("Building FAISS index...")
faiss_index, _ = build_index(chunks, embed_model)
print("Index ready!")

# ── 8. CHATBOT FUNCTION ────────────────────────────────────
def chatbot(user_query, history):
    try:
        retrieved = retrieve(user_query, embed_model, faiss_index, chunks)
        answer = generate_answer(user_query, retrieved)
        sources = list(set([c["source"] for c in retrieved]))
        answer += f"\n\n📚 Sources: {', '.join(sources)}"
        return answer
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ── 9. GRADIO UI ───────────────────────────────────────────
demo = gr.ChatInterface(
    fn=chatbot,
    title="📖 RAG Chatbot",
    description="Ask questions based on the uploaded documents.",
)
demo.launch()
