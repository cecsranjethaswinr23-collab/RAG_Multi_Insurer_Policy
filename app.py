import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from RAG_LLM_Promot import llm_prompt

# 1. Load environment variables from .env
load_dotenv()

st.set_page_config(
    page_title="Multi-Insurer Policy Assistant",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Multi-Insurer Health Policy Assistant")
st.caption("Compare quantitative sublimits, coverage terms, and exclusions across policy providers.")

# 2. Cache resources so FAISS loads into memory only once
@st.cache_resource
def init_rag_system():
    # MUST match the exact embedding model used in ingest.ipynb
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Load saved FAISS store from disk
    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    # Initialize Gemini Flash LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=0.1  # Low temperature for strict factual accuracy
    )
    
    return vectorstore, llm

try:
    vectorstore, llm = init_rag_system()
    st.sidebar.success("✅ FAISS Vector Index Loaded")
except Exception as e:
    st.error(f"Failed to load vector store: {e}. Did you run `ingest.ipynb` first?")
    st.stop()

# 3. Sidebar Filtering Options
st.sidebar.header("Options")
company_filter = st.sidebar.selectbox(
    "Filter by Provider:",
    ["All Insurers", "Star Health", "Care Health", "HDFC ERGO"] # Customize based on your PDFs
)


# 5. User Input
user_query = st.text_input("Ask a question about coverage, sublimits, or exclusions:")

if user_query:
    with st.spinner("Searching policy documents..."):
        # Configure search metadata filter if selected
        search_kwargs = {"k": 4}
        if company_filter != "All Insurers":
            search_kwargs["filter"] = {"insurer": company_filter}

        # Step A: Retrieve matching chunks from local FAISS
        retrieved_docs = vectorstore.similarity_search(user_query, **search_kwargs)
        
        # Step B: Format context string
        context_str = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # Step C: Send context + query to Gemini
        chain = llm_prompt | llm
        response = chain.invoke({"context": context_str, "question": user_query})

    # 6. Display Response
    st.markdown("### 🤖 Answer")
    st.write(response.content)

    # 7. Expandable Source Citations (Recruiter Delight)
    with st.expander("🔍 View Retrieved Context Chunks & Page Numbers"):
        for i, doc in enumerate(retrieved_docs, start=1):
            st.markdown(f"**Chunk {i} | Insurer:** `{doc.metadata.get('insurer')}` | **Source:** `{doc.metadata.get('source')}` (Page {doc.metadata.get('page')})")
            st.code(doc.page_content, language="markdown")