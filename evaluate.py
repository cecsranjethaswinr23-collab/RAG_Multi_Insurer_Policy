import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

# Import your LangChain components
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


# the seperate llm prompt file
from RAG_LLM_Prompt import llm_prompt

def run_evaluation():
    # 1. Initialize your RAG System
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.1)


    # THE TEMPLATE FOR THE PROMPT TO PUT IT IN A CHAIN  

    #   prompt_template = """Use the following pieces of context to answer the question at the end. 
    #   Context: {context}
    #   Question: {question}
    #   Answer:"""
    #   llm_prompt = PromptTemplate.from_template(prompt_template)


    chain = llm_prompt | llm

    # 2. Define test cases
    eval_questions = [
        "What are the exclusions for pre-existing conditions?",
        "Does the policy cover maternity expenses?"
    ]
    ground_truths = [
        "Pre-existing conditions are excluded for the first 24 months of the policy.",
        "Yes, maternity expenses are covered up to a limit of $5,000 after a 9-month waiting period."
    ]

    answers = []
    contexts = []

    print("Running queries through FAISS...")
    # 3. Generate answers and retrieve contexts
    for q in eval_questions:
        retrieved_docs = vectorstore.similarity_search(q, k=4)
        context_string = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        
        response = chain.invoke({"context": context_string, "question": q})
        answer_text = response.content[0]["text"] if isinstance(response.content, list) else response.content
        
        answers.append(answer_text)
        contexts.append([doc.page_content for doc in retrieved_docs])

    # 4. Format the dataset for Ragas
    dataset = Dataset.from_dict({
        "question": eval_questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    print("Scoring responses with Ragas...")
    # 5. Run the evaluation
    results = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
    df_results = results.to_pandas()
    
    # 6. Save and print results
    df_results.to_csv("evaluation_results.csv", index=False)
    print("\nEvaluation complete! Results saved to evaluation_results.csv")
    
    # Print average scores to the terminal
    print("\n--- Average Scores ---")
    print(df_results[['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']].mean())

if __name__ == "__main__":
    run_evaluation()

def evaluate_single_query(user_query, ground_truth, company_filter):

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") # embedding model
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) # loading local FAISS index file
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.1) # setting up the LLM to use for the evaluation
    
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
    
    print("Scoring with Ragas...")
    results = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
    print(f"the actual results:\n {results}\n\n")
    df_results = results.to_pandas()
    
    print("Single Evaluation Results\n")
    print(f"Question: {user_query}")
    print(f"Answer:   {answer_text}")
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
    evaluate_single_query(
        user_query,
        ground_truth,
        company_filter 
    )