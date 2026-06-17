# Task 3 — RAG Chatbot

## What is RAG?
Retrieval-Augmented Generation (RAG) retrieves relevant chunks from documents
and uses an LLM to generate answers based on that context.

## How It Works
1. Documents are split into chunks
2. Each chunk is embedded using Sentence Transformers (all-MiniLM-L6-v2)
3. Embeddings are stored in a FAISS vector index
4. User query is embedded and matched against the index
5. Top 3 relevant chunks are retrieved
6. Retrieved context + query is passed to TinyLlama LLM
7. LLM generates the final answer with source citation

## Tech Stack
- Sentence Transformers — embeddings
- FAISS — vector similarity search
- TinyLlama — local LLM (no API key needed)
- Gradio — chat UI

## How to Run
pip install sentence-transformers faiss-cpu transformers torch gradio numpy
python rag_chatbot.py