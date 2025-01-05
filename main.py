import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
import requests
from streamlit_lottie import st_lottie
from dotenv import load_dotenv

load_dotenv()

## Load the GROQ API KEY 
groq_api_key = os.getenv('GROQ_API_KEY')

# Load Lottie animation
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_coding = load_lottieurl("https://lottie.host/687b2922-0663-44f3-bce9-ea6e18556a99/yGEWjw5o7z.json")

# Custom Styling
st.markdown(
    """
    <style>
    body {
        background-color: #121212;
        color: #FFFFFF;
        font-family: 'Arial', sans-serif;
    }
    .st-emotion-cache-yw8pof {
        width: 100%;
        padding: 6rem 0rem 10rem;
        max-width: 75rem;
    }
    .stApp {
        padding: 20px;
        border-radius: 10px;
    }
    .stTextInput>div>div>input {
        color: black;
        background-color: white;
        border: 1px solid #444444;
    }
    .stTextInput>div>div>input::placeholder {
        color: #BBBBBB;
    }
    .stMarkdown {
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Container with two columns
with st.container():
    left_column, right_column = st.columns(2)
    with left_column:
        st.title("Gokul's Assistant")
        # Input box under title
        prompt1 = st.text_input("Want to know more about Gokul? Ask me!!")

    with right_column:
        st_lottie(lottie_coding, speed=1, width=400, height=400, key="coding")

# Initialize the Llama3 model
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="Llama3-8b-8192"
)

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please respond in a concise format suitable for a chatbot.
    Example responses:
    Question: Does he have skill in AWS?
    Answer: Yes, he is skilled in AWS.

    Question: Who are you?
    Answer: I'm Gokul's Assistant. How can I help you?

    Question: What are his career goals?
    Answer: His career goals are to leverage technical expertise in Python, SQL, and AWS to build scalable data pipelines, develop AI-driven solutions for predictive analytics, and lead cross-functional teams in delivering impactful, data-driven strategies.

    Question: Tell me about Gokul Ramesh
    Answer: An Innovative data science graduate student with a strong foundation in data analysis, machine learning, and software development. Adept at delivering scalable, user-centric solutions through advanced analytical techniques and robust programming skills. With professional experience in front-end development and data-driven applications at Infosys, he has successfully led impactful projects such as a Pre-Order web application for Toyota, improving operational efficiency by 95%. His expertise spans Python, SQL, AWS, and cutting-edge visualization tools, making him a dynamic problem-solver and a valuable asset in data-driven environments.

    Default Response: I don't have information about that.
     
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

def vector_embedding():
    if "vectors" not in st.session_state:
        try:
            # Initialize embeddings (HuggingFace instead of OpenAI)
            st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

            # Load PDF documents
            st.session_state.loader = PyPDFDirectoryLoader("./Docs")
            st.session_state.docs = st.session_state.loader.load()

            # Text splitting
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(
                st.session_state.docs
            )

            # Generate vectors
            st.session_state.vectors = FAISS.from_documents(
                st.session_state.final_documents,
                st.session_state.embeddings
            )

        except Exception as e:
            st.error(f"Error in vector embedding: {str(e)}")

# Automatically load documents and create vectors when the app starts
vector_embedding()

import time

# Response displayed immediately below input box
if prompt1:
    try:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})

        # Display the concise answer directly below input box
        with left_column:
            st.write(response['answer'])

    except Exception as e:
        st.error(f"Error in response generation: {str(e)}")

