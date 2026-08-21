import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

# Import your LangChain components
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


import os
from dotenv import load_dotenv

load_dotenv()  # Loads GOOGLE_API_KEY


# the seperate llm prompt file
from RAG_LLM_Prompt import llm_prompt


def evaluate_single_query(user_query, ground_truth, company_filter):

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") # embedding model
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) # loading local FAISS index file
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.1) # setting up the LLM to use for the evaluation

    wrap_llm = LangchainLLMWrapper(llm) # wrapper used to explicitly mention the llm
    wrap_embeddings = LangchainEmbeddingsWrapper(embeddings) # wrapper used to explicitly mention the embeddings model
    
    chain = llm_prompt | llm


    print(f"Searching '{company_filter}' documents for: '{user_query}'...\n\n\n")
    
    # Apply metadata filter identically to the UI logic
    search_kwargs = {"k": 4}
    if company_filter != "All Insurers":
        search_kwargs["filter"] = {"insurer": company_filter}

    retrieved_docs = vectorstore.similarity_search(user_query, **search_kwargs)
    context_string = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
    
    response = chain.invoke({"context": context_string, "question": user_query})
    answer_text = response.content[0]["text"] if isinstance(response.content, list) else response.content

    # Ragas expects arrays, so wrap the single strings/objects in lists
    data = {
        "question": [user_query],
        "answer": [answer_text],
        "contexts": [[doc.page_content for doc in retrieved_docs]],
        "ground_truth": [ground_truth]
    }
    dataset = Dataset.from_dict(data)

    #----------------------------------------------------------------------------------------------------------------------------
    # evaluation starts here
    print("Scoring with Ragas...")
    results = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall],llm=wrap_llm,embeddings=wrap_embeddings)
    print(f"the actual results:\n {results}\n\n")
    df_results = results.to_pandas()
    
    print("Single Evaluation Results\n")
    print(f"Question: {user_query}") # the query to the RAG application
    print(f"Answer:   {answer_text}") # the response from the llm
    print("\nScores:")
    print(f"- Faithfulness:      {df_results.iloc[0]['faithfulness']:.4f}")
    print(f"- Answer Relevancy:  {df_results.iloc[0]['answer_relevancy']:.4f}")
    print(f"- Context Precision: {df_results.iloc[0]['context_precision']:.4f}")
    print(f"- Context Recall:    {df_results.iloc[0]['context_recall']:.4f}")

# user query
user_query="what are the hazardous activities that will not be coverd by star insurance?"

# the filtering function added to specifically take the chunks only with the [metadata][insurer:name]
# manually selecting it from the list all_the_insurance_names mentioned above
# all the insrance names to specifically take info from it
""" 
the below list is the names of all the metadata of the names 

"""
all_the_insurance_names=["All Insurers",
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

company_filter="All Insurers"

# the ground truth to check the groundedness
ground_truth="Expenses related to any treatment necessitated due to participation as a professional in " \
"hazardous or adventure sports, including but not limited to, para-jumping, rock climbing, mountaineering, " \
"rafting, motor racing, horse racing or scuba diving, hand gliding, sky diving, deep-sea diving."


if __name__ == "__main__":
    # Execute a single test directly in PowerShell
    evaluate_single_query(user_query,ground_truth,company_filter)