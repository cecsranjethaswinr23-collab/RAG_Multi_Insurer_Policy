# RAG_Multi_Insurer_Policy



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
       │             Online Phase : preprocessing.ipynb          │
       │                                                         │
       │                                                ▼        │
       │  User Question ──► Streamlit UI ──► FAISS.load_local()  │
       │                                               │         │
       │   LLM Response  ◄── Prompt + Context ◄────────┘         │
       └─────────────────────────────────────────────────────────┘