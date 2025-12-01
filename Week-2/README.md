🧠 Simple Resume Analyzer: RAG Chatbot

This project deploys a Retrieval-Augmented Generation (RAG) chatbot using Streamlit, LangChain, Qdrant, and Groq. Its primary function is to analyze a given PDF resume and answer user questions based only on the content found within that resume.

The application is built to be simple and stable, using Streamlit's resource caching (@st.cache_resource) to handle heavy model loading efficiently.

✨ Features

PDF Analysis: Loads and processes a local PDF file (Archit_Dadhich's_Resume.pdf).

Vector Database (Qdrant): Stores document chunks as vector embeddings for fast and relevant search. 

Groq Inference: Uses the ultra-fast Groq platform and the Llama 3.1 model to generate natural-language answers.

Streamlit UI: Provides a simple, interactive web interface for asking questions.

Performance: Uses caching to ensure the application remains stable and responsive even with frequent user interactions.

🛠️ Prerequisites

Before running this application, you need to set up two things:

1. Groq API Key

The application requires a Groq API key to use the Llama language model.

Get your API key from the Groq Console.

Set it as an environment variable in your terminal before running the app:

export GROQ_API_KEY="your-api-key-here"


2. Qdrant Vector Database

Qdrant runs most reliably in a Docker container.

Ensure you have Docker installed.

Run the Qdrant server, exposing the necessary port (6333):

docker run -p 6333:6333 qdrant/qdrant


🚀 Setup and Installation

Follow these steps to get the project running locally.

1. Clone or Download the Code

Ensure your app.py file is in a local directory.

2. Prepare the Resume

Place the PDF file you want to analyze in the same directory as app.py. The code is currently set to look for a file named:

./Archit_Dadhich's_Resume.pdf


If your file has a different name, you must update the PDF_PATH variable in app.py.

3. Install Dependencies

You'll need streamlit, the langchain packages, groq, and qdrant-client.

pip install streamlit langchain langchain-groq langchain-huggingface qdrant-client pypdf


4. Run the Application

With your GROQ_API_KEY set and the Qdrant Docker container running, start the Streamlit app:

streamlit run app.py


Streamlit will open the application in your web browser.

💡 How the Code Works (The RAG Flow)

The application follows three simple, sequential steps every time you click the "Get Response" button:

Retrieval Step:

The user's user_query is converted into a vector by the embeddings model.

The retriever searches the qdrant database for the top 5 most similar text chunks from the resume.

Augmentation Step:

The retrieved text chunks (context) and the user_query are combined into a final, highly structured message using the prompt_template. This message instructs the LLM to only use the provided context.

Generation Step:

The complete message (final_prompt) is sent to the llm (Groq/Llama 3.1).

The LLM reads the context and the question, generates a concise answer, and the result is displayed in Streamlit.