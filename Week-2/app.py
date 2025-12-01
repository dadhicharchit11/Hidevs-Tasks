from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate # We keep the prompt for good structure
import streamlit as st
import os


# --- STEP 1: CACHING HEAVY SETUP (The most important part for performance) ---
# The '@st.cache_resource' decorator is necessary! 
# It tells Streamlit to run this function ONLY ONCE when the app starts.
# This prevents the heavy PyTorch embedding model from reloading every time you click a button, 
# which fixes errors and keeps the app fast.
@st.cache_resource
def setup_rag_pipeline(pdf_path, url, collection_name):
    """Initializes the database and embedding model—runs only one time."""
    
    # 1a. Load the PDF file and break it into smaller pieces (chunks)
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print("Document chunking is complete.")
    
    # 1b. Create the Embedding Model (A model that turns text into numbers/vectors)
    model_name = "BAAI/bge-small-en"
    # We explicitly tell the model to use the CPU for stability
    model_kwargs = {'device': 'cpu'} 
    encode_kwargs = {'normalize_embeddings': True}
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    print("Embedding model loaded.")

    # 1c. Create/Connect to Qdrant (Our vector database)
    # This process embeds the text chunks and stores them in Qdrant
    qdrant = QdrantVectorStore.from_documents(
        texts,
        embeddings,
        url=url, 
        collection_name=collection_name,
        # force_recreate=True 
    )
    print(f"Qdrant Collection '{collection_name}' is ready!")

    # We return the ready-to-use database object
    return qdrant


# --- STEP 2: CONFIGURATION & INITIALIZATION ---
PDF_PATH = "./Archit_Dadhich's_Resume.pdf"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "vector_db"

# Call the setup function to get the initialized database object
qdrant = setup_rag_pipeline(PDF_PATH, QDRANT_URL, COLLECTION_NAME)


# Create the Language Model (LLM) object from Groq (This is lightweight and runs every time)
llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.1-8b-instant")
print("Groq LLM is ready.")

# Create the Retriever (This tool finds the most relevant documents in the database)
retriever = qdrant.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Define the Prompt Template (The instructions we give to the LLM)
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an intelligent assistant tasked with answering user queries based on provided context. Use the following context to respond to the user's question. If the context does not contain the answer, state that you cannot answer based on the resume."),
    ("human", "Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:")
])


# --- STEP 3: STREAMLIT FRONTEND AND SIMPLIFIED EXECUTION ---
st.title("🧠 Simple Resume Analyzer (Groq + Qdrant)")
st.caption("Ask questions about the content of the PDF resume.")

# Input box for the user's question
user_query = st.text_input("Enter your question here:")


# When the user clicks the button, the RAG steps run sequentially
if st.button("Get Response", type="primary"): 
    if user_query: # Only run if the user typed something
        with st.spinner("🚀 Analyzing resume content..."): 
            try:
                # 1. RETRIEVAL STEP: Find relevant text chunks using the retriever
                retrieved_docs = retriever.invoke(user_query)
                
                # Combine the text content from the documents into one string
                context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                
                # 2. PROMPT STEP: Insert the context and the user's question into the template
                final_prompt = prompt_template.format(context=context, query=user_query)
                
                # 3. GENERATION STEP: Send the final prompt to the LLM
                llm_response = llm.invoke(final_prompt)
                
                # The final answer is in the 'content' attribute of the response object
                final_answer = llm_response.content
                
                st.success("✅ Response:")
                st.markdown(final_answer) 
                
            except Exception as e:
                # If an error happens, show the user a helpful message
                st.error(f"An error occurred: {e}")
                st.info("Please ensure your Qdrant container is running and your GROQ_API_KEY is set correctly.")
    else:
        st.warning("Please enter a question to analyze the resume.")