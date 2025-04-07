import streamlit as st
import os
import sys
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create directories
os.makedirs("./model_cache", exist_ok=True)
os.makedirs("./vector_store", exist_ok=True)

# Set environment variables for better deployment compatibility
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU
os.environ["USE_TORCH"] = "0"  # Disable torch for certain operations
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable parallelism in tokenizers

# Load environment variables
load_dotenv()

# Add a Streamlit error handling decorator
def streamlit_error_handler(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Please try refreshing the page or contact support if the issue persists.")
            # Show more detailed error in the logs
            logger.exception("Detailed error information:")
            return None
    return wrapper

# Initialize Streamlit app
st.set_page_config(
    page_title="Gokul's Assistant",
    page_icon="👨‍💻",
    layout="wide"
)

# Custom CSS
st.markdown(
    """
    <style>
    body {
        background-color: #121212;
        color: #FFFFFF;
        font-family: 'Arial', sans-serif;
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

# Now import the rest of the dependencies
# If any import fails, it will be caught by the decorator
try:
    import requests
    from streamlit_lottie import st_lottie
    from langchain_groq import ChatGroq
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate
    from langchain.chains import create_retrieval_chain
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.embeddings import FakeEmbeddings
    from langchain_community.document_loaders import PyPDFLoader
except ImportError as e:
    st.error(f"Failed to import required dependencies: {str(e)}")
    st.info("Try installing the required packages with: pip install -r requirements.txt")
    st.stop()

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'last_processed_message' not in st.session_state:
    st.session_state.last_processed_message = None
if 'input_key' not in st.session_state:
    st.session_state.input_key = 0

def increment_input_key():
    st.session_state.input_key += 1

# Check for required environment variables
groq_api_key = os.getenv('GROQ_API_KEY')
if not groq_api_key:
    st.error("GROQ_API_KEY not found in environment variables. Please add it to your .env file or Streamlit secrets.")
    st.stop()

# Load Lottie animation
@streamlit_error_handler
def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception as e:
        logger.error(f"Error loading animation: {str(e)}")
        return None

lottie_coding = load_lottieurl("https://lottie.host/687b2922-0663-44f3-bce9-ea6e18556a99/yGEWjw5o7z.json")

@st.cache_resource(show_spinner=False)
def initialize_components():
    try:
        # Initialize LLM
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="Llama3-8b-8192",
            temperature=0.5,
            max_tokens=4096
        )
        
        # Initialize embeddings with special handling for deployment environments
        embeddings = None
        
        # Try multiple embedding strategies
        embedding_strategies = [
            # Strategy 1: Use simpler model first for deployment
            lambda: HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
                model_kwargs={'device': 'cpu'},
                cache_folder="./model_cache"
            ),
            # Strategy 2: Use the main model
            lambda: HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2", 
                model_kwargs={'device': 'cpu'},
                cache_folder="./model_cache"
            ),
            # Strategy 3: Use fake embeddings as last resort
            lambda: FakeEmbeddings(size=384)
        ]
        
        # Try each strategy in order
        for i, strategy_fn in enumerate(embedding_strategies):
            try:
                embeddings = strategy_fn()
                logger.info(f"Successfully initialized embeddings with strategy {i+1}")
                break
            except Exception as e:
                logger.warning(f"Embedding strategy {i+1} failed: {str(e)}")
                continue
        
        if embeddings is None:
            raise Exception("All embedding strategies failed")
        
        # Check if we have a pre-saved vector store
        vector_store_path = "./vector_store"
        
        if os.path.exists(vector_store_path) and os.listdir(vector_store_path):
            try:
                # Try to load the vector store from disk
                vectors = FAISS.load_local(vector_store_path, embeddings)
                logger.info(f"Successfully loaded vector store from {vector_store_path}")
                return llm, vectors
            except Exception as e:
                logger.warning(f"Failed to load vector store from disk: {str(e)}. Will recreate.")
                # Continue with creating a new vector store
        
        # Load documents
        documents = []
        
        # Main PDF
        pdf_path = "./Docs/chatbot.pdf"
        if os.path.exists(pdf_path):
            try:
                pdf_loader = PyPDFLoader(pdf_path)
                pdf_docs = pdf_loader.load()
                # Add metadata to track source
                for doc in pdf_docs:
                    doc.metadata["source"] = "chatbot.pdf"
                documents.extend(pdf_docs)
                logger.info(f"Loaded PDF from {pdf_path} - {len(pdf_docs)} pages")
            except Exception as pdf_error:
                logger.error(f"Error loading PDF: {str(pdf_error)}")
        
        if not documents:
            raise Exception("No documents found to process")
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Smaller chunks for better performance
            chunk_overlap=200,
            length_function=len,
        )
        
        chunks = text_splitter.split_documents(documents)
        
        # Create vector store
        vectors = FAISS.from_documents(chunks, embeddings)
        
        # Save the vector store to disk for future use
        try:
            vectors.save_local(vector_store_path)
            logger.info(f"Successfully saved vector store to {vector_store_path}")
        except Exception as e:
            logger.warning(f"Failed to save vector store to disk: {str(e)}")
        
        logger.info(f"Successfully processed {len(documents)} documents with {len(chunks)} chunks")
        return llm, vectors
    
    except Exception as e:
        logger.error(f"Error in initialization: {str(e)}")
        st.error(f"Error initializing components: {str(e)}")
        return None, None

# Initialize components with loading message
with st.spinner("Loading information..."):
    try:
        llm, vectors = initialize_components()
    except Exception as init_error:
        st.error(f"Failed to initialize components: {str(init_error)}")
        llm, vectors = None, None

# Show a warning if components failed to initialize
if not llm or not vectors:
    st.warning("Some components failed to initialize. You can still try using the application, but functionality may be limited.")

# Container with two columns
with st.container():
    left_column, right_column = st.columns(2)
    with left_column:
        st.title("Gokul's Assistant")
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown("---")
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f"**You:** {message['content']}")
                else:
                    st.markdown(f"**Assistant:** {message['content']}")
                st.markdown("---")
            
            if st.button("Clear Chat History"):
                st.session_state.chat_history = []
                st.session_state.last_processed_message = None
                increment_input_key()
                st.rerun()
        
        user_input = st.text_input(
            "Ask me about Gokul:", 
            key=f"input_field_{st.session_state.input_key}"
        )

    with right_column:
        if lottie_coding:
            st_lottie(lottie_coding, speed=1, width=400, height=400, key="coding")
        else:
            st.image("https://via.placeholder.com/400x400.png?text=Gokul's+Assistant", width=400)

@streamlit_error_handler
def process_input(user_input):
    if not llm or not vectors:
        return "Sorry, the assistant is not fully initialized. Please refresh the page and try again."

    st.session_state.last_processed_message = user_input
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Create prompt template
    prompt = ChatPromptTemplate.from_template(
        """You are Gokul's personal AI assistant. Use the following information about Gokul to answer questions.

        Information about Gokul:
        {context}

        IMPORTANT BACKGROUND INFORMATION ABOUT GOKUL (Include this information only when directly relevant):

        SKILLS:
        - Programming Languages: Python, JavaScript, Java, SQL
        - Machine Learning: Regression, Classification, Random Forest, XGBoost, Feature Engineering
        - Data Analysis: Data Cleaning, Visualization, Statistical Analysis
        - Tools: Git, AWS, Docker, Jupyter, PyTorch, TensorFlow

        Human Question: {input}

        Instructions:
        - ONLY use information from chatbot.pdf for your answers
        - Be comprehensive and include ALL relevant details from the retrieved information
        - If the information isn't in the retrieved documents, say so clearly
        - Present information in a clear, friendly and conversational way
        - Use bullet points for structured information when appropriate

        Answer:"""
    )
    
    # Create retrieval chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectors.as_retriever(search_kwargs={"k": 8})
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    with st.spinner("Thinking..."):
        response = retrieval_chain.invoke({"input": user_input})
        answer = response['answer'].strip()
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    return answer

if user_input and user_input != st.session_state.last_processed_message:
    try:
        process_input(user_input)
        increment_input_key()
        st.rerun()
    except Exception as e:
        st.error(f"Error processing your request: {str(e)}")
        logger.exception("Error processing request:") 