# 🤖  RAG_Multi_Insurer_Policy
An intelligent, aware Retrieval-Augmented Generation (RAG) system designed to search, compare, and answer queries across multiple insurance providers with source citations of multiple insurance policies.

## 🌟 Key Features
**Structure-Aware Document Ingestion:** Uses pymupdf4llm to convert policy PDFs into structured Markdown, preserving critical tables, coverage sub-limits, and clause headers.

**Automated Metadata Tagging:** Automatically scans nested folder structures (data/policies/<Insurer>/) to tag chunks with provider names, document titles, and page numbers.

Fast Vector Search (FAISS): Embeds text chunks locally using sentence-transformers and indexes them in a local FAISS vector database.

Dynamic Provider Filtering: Allows users to query all policies simultaneously or narrow searches down to specific insurance providers (e.g., Care, HDFC, Star, United India) directly in the Streamlit UI.

Grounded LLM Responses: Powered by Google Gemini via LangChain (LCEL) to produce precise, hallucination-free answers formatted in clean Markdown.

Transparent Citations: Provides exact source document names and page numbers for every answer generated.

## 🛠️ Tech Stack & Tools

**Language:** Python
**Framework:** Langchain
**VectorDB:** FAISS (Facebook AI Similarity Search)
**Frontend:** Streamlit
**LLM:** Gemini (Google)
**Cloud Deployment:** HuggingFace Spaces

## 📐 Architecture & Pipeline Flow
       ┌─────────────────────────────────────────────────────────┐
       │            Offline Phase : preprocessing.ipynb          │
       │                                                         │
       │                                                         │
       │  PDF ──► Docling/pdfplumber ──► Chunks ──► Embeddings   │
       │                                                │        │
       └────────────────────────────────────────────────┼────────┘
                                                        ▼
                                             [ faiss_index/ Folder ]
                                             ├── index.faiss
                                             └── index.pkl
                                                        │
       ┌────────────────────────────────────────────────┼────────┐
       │             Online Phase : app.py                       │
       │                                                         │
       │                                                ▼        │
       │  User Question ──► Streamlit UI ──► FAISS.load_local()  │
       │                                               │         │
       │   LLM Response  ◄── Prompt + Context ◄────────┘         │
       └─────────────────────────────────────────────────────────┘