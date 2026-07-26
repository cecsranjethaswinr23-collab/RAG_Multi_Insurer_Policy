import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from RAG_LLM_Prompt import llm_prompt

# dependancies
#---------------------------------------------------------------------------------------------------------->

load_dotenv() #Load environment variables from .env

# webpage name and icon
st.set_page_config(
    page_title="PolicyLens_AI",
    page_icon="💎",
    layout="wide"
)

# Sidebar
# About title
st.sidebar.title("About")

# sidebar markdown 1
st.sidebar.markdown("""
### 🎯 WHAT IT DOES:
This Retrieval-Augmented Generation (RAG) application search and answer queries across multiple insurance policies documents seemlessly. 
The system retrieves relevant context while enabling dynamic provider-level filtering through an interactive Streamlit dashboard, and
it has the data of multiple Insurance companies health insurance policies and a company's motor vehicle insurance policies.
""")

# sidebar markdown 2
companies_name=[
    "Care (Health Insurance) ❤️",
    "HDFC (Health Insurance) 🩺",
    "Star (Health Insurance) 💊",
    "United India(vehicle insurance) 🚗&🏍️ as UI"
    ]
st.sidebar.markdown("""
#### 🫠 PROCEDURE:
You can filter the specific insurance you want to question about or else conclusively fetch information from all the insurance policies 
data to get the the information you need. Before using the Application refer the policies and their wordings to understand how this 
application can be questioned to get right responses for you. The policies of insurance companies are
""")
for company in companies_name:
    st.sidebar.markdown(f"- ***{company}***")

# Google Drive link with policies for reference
st.sidebar.link_button("💻 Go to Drive", "https://drive.google.com/drive/folders/1Rk2pvMNtB_v-P5YPo-qjoss2xBgO1VeQ?usp=sharing", use_container_width=True)

# tech used in the application
st.sidebar.subheader("🛠️ Tech Stack")
tech_stack = [
    "Python",
    "Streamlit",
    "Langchain",
    "FAISS",
    "Gemini AI"
]
for tech in tech_stack:
    st.sidebar.markdown(f"- **{tech}**")

# github link bar
st.sidebar.subheader("🔗 Source Code")
st.sidebar.link_button("💻 Go to GitHub Repository", "https://github.com/cecsranjethaswinr23-collab/RAG_Multi_Insurer_Policy", use_container_width=True)

# author bar
st.sidebar.markdown("### 👨‍💻 Developed By")
st.sidebar.markdown("**Ranjeth Aswin Ravindran**")
st.sidebar.caption("Data Scientist & AI Engineer")

# Email bar
st.sidebar.markdown("""
📧 **Contact:** cecsranjethaswinr23@gmail.com
""")
# end of about section
# ------------------------------------------------------------------------------------------------------------------------------------


# main page

st.title("🤖  PolicyLens AI (RAG)")
st.subheader("Policy Info retrieval AI application 🌐")
st.markdown("""The application retrieves the insurance policy informations accurately from the data present in the vector database, 
the policies that are in the database can be accessed through the link in the ABOUT section.
""")
st.markdown("""Please refer the policies to ask the questions regarding the policies to the RAG appllication""")
st.caption("Compare quantitative sublimits, coverage terms, and exclusions across policy providers.")


st.header("👉Company Policies👈")

company_filter = st.selectbox(
    "Filter by Provider:",
["All Insurers",
 'Care Senior Health',
 'Care Supreme',
 'Care Ultimate Joy',
 'Hdfc Equicover Health',
 'Hdfc Pradhan Mantri Suraksha Bima Yojana',
 'Hdfc Women Suraksha',
 'Star Comprehensive Policy',
 'Star Medi Classic Insurance Policy',
 'Star Senior Citizen Policy',
 'Star Women Policy',
 'Ui Commercial Vehicles 3Rd Party',
 'Ui Commercial Vehicles Cover',
 'Ui Private Bike 1Yr Od 3Yr Tp',
 'Ui Private Bike 1Yr Od 5Yr Tp',
 'Ui Private Bike Only Tp',
 'Ui Private Car Only Tp',
 'Ui Scooter Own Damage']
)


def rag_system():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") # embedding model
    vectorstore = FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True) # loading saved FAISS
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash",temperature=0.1)
    
    return vectorstore, llm

try:
    vectorstore, llm = rag_system()
    st.sidebar.success("✅ FAISS Vector Index Loaded")
except Exception as e:
    st.error(f"Failed to load vector store: {e}. Did you run `ingest.ipynb` first?")
    st.stop()

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
    st.markdown("### 🤖 Response")
    answer_text = response.content[0]["text"] if isinstance(response.content, list) else response.content
    st.markdown(answer_text)


    # 7. Expandable Source Citations (Recruiter Delight)
    with st.expander("🔍 View Retrieved Context Chunks & Page Numbers"):
        for i, doc in enumerate(retrieved_docs, start=1):
            st.markdown(f"**Chunk {i} | Insurer:** `{doc.metadata.get('insurer')}` | **Source:** `{doc.metadata.get('source')}` (Page {doc.metadata.get('page')})")
            st.code(doc.page_content, language="markdown")