# AI PDF Chatbot

A local PDF chatbot built with Streamlit, LangChain, FAISS, Hugging Face embeddings, and Ollama (Llama 3.2).

## Features
- Upload any PDF
- Extract text from the PDF
- Split text into chunks
- Create embeddings with Hugging Face
- Store chunks in FAISS
- Ask questions about the PDF
- Get summaries of papers
- Show source chunks with page numbers
- Display extracted paper title

## Tech Stack
- Python
- Streamlit
- PyMuPDF
- LangChain
- FAISS
- sentence-transformers
- Ollama
- Llama 3.2

## Project Structure
```text
pdf-rag-chatbot/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── screenshots/
└── utils/
    ├── pdf_loader.py
    ├── text_splitter.py
    ├── embedding.py
    ├── vector_store.py
    ├── retriever.py
    └── rag_chain.py