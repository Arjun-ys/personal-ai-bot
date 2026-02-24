import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
import tempfile
import fitz  # This is PyMuPDF
from rapidocr_onnxruntime import RapidOCR
from langchain_community.chat_models import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.memory import VectorStoreRetrieverMemory
from langchain_classic.chains import ConversationChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOllama
# --- CONFIGURATION ---
PAGE_TITLE = "My Offline AI Agent"
MODEL_NAME = "llama3"  # Make sure you have run 'ollama run llama3' in your terminal previously
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
PERSIST_DIR = "./ai_memory_db"

st.set_page_config(page_title=PAGE_TITLE, layout="wide")
st.title(PAGE_TITLE)

# --- 1. INITIALIZE RESOURCES (Cached for performance) ---

@st.cache_resource
def get_embeddings():
    # Downloads the model locally to your machine (runs on CPU/GPU)
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

@st.cache_resource
def get_vector_store():
    # Connects to the local ChromaDB database
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name="personal_chat_history"
    )

vectordb = get_vector_store()

# --- 2. SIDEBAR: KNOWLEDGE INGESTION ---

st.sidebar.header("🧠 Knowledge Base")
st.sidebar.write("Upload documents (PDF, TXT, LOG) to give your AI long-term memory.")

uploaded_file = st.sidebar.file_uploader("Choose a file...", type=['pdf', 'txt', 'log'])

# Helper function for OCR (Optical Character Recognition)
def get_text_from_scanned_pdf(pdf_path):
    ocr = RapidOCR()
    doc = fitz.open(pdf_path)
    full_text = ""
    
    # Loop through every page in the PDF
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # 1. Convert the page to an image (Pixmap)
        pix = page.get_pixmap()
        img_bytes = pix.tobytes("png")
        
        # 2. Run AI recognition on the image
        result, _ = ocr(img_bytes)
        
        # 3. Extract the text if found
        if result:
            for line in result:
                full_text += line[1] + "\n"
                
    return full_text

if uploaded_file is not None:
    with st.spinner("Processing... (If scanned, this may take a moment)"):
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            documents = []
            
            # CASE A: It's a PDF
            if uploaded_file.name.endswith('.pdf'):
                # First, try the standard fast loader
                loader = PyPDFLoader(tmp_file_path)
                documents = loader.load()
                
                # CHECK: Did we get empty text? (Indicates it's a scan)
                if not documents or len(documents[0].page_content.strip()) < 10:
                    st.sidebar.warning("Scanned document detected! Switching to AI OCR (Slower)...")
                    
                    # Run our custom OCR function
                    scanned_text = get_text_from_scanned_pdf(tmp_file_path)
                    
                    # Manually wrap the text into a Document format for LangChain
                    from langchain_core.documents import Document
                    documents = [Document(page_content=scanned_text, metadata={"source": uploaded_file.name})]

            # CASE B: Text or Log file
            else:
                loader = TextLoader(tmp_file_path, encoding='utf-8')
                documents = loader.load()

            # Split text and save to memory
            if documents and len(documents[0].page_content) > 0:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_documents(documents)
                vectordb.add_documents(chunks)
                st.sidebar.success(f"✅ Memorized {len(chunks)} chunks from {uploaded_file.name}!")
            else:
                st.sidebar.error("Could not read any text from this file.")
        
        except Exception as e:
            st.sidebar.error(f"Error processing file: {e}")
        
        finally:
            # Clean up
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

# --- 3. SET UP THE BRAIN (LLM + MEMORY) ---

# Configure Retrieval: Fetch top 3 relevant past memories
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Model Settings")
# Create the toggle switch in the UI
llm_choice = st.sidebar.radio(
    "Select your AI Brain:", 
    ["Local Offline (Llama 3)", "Cloud API (Gemini 3.1 Pro)"]
)

# Initialize the chosen model dynamically
if llm_choice == "Local Offline (Llama 3)":
    try:
        llm = ChatOllama(model="llama3")
    except Exception as e:
        st.error("Make sure Ollama is running in your terminal!")
        st.stop()
else:
    # Using the latest Gemini 3.1 Pro preview model
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview")

# Configure Retrieval Memory
retriever = vectordb.as_retriever(search_kwargs={"k": 3})
memory = VectorStoreRetrieverMemory(retriever=retriever)

# The optimal role-aware chat prompt for Llama 3 / Gemini
chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are a highly capable personal AI assistant.\n"
        "You have access to a long-term memory database of past conversations and documents.\n"
        "Your task is to provide helpful, accurate, and contextually relevant responses to the user's queries.\n"
        "Be sure to learn about the user's preferences and needs over time, and use that information to improve your responses, while being mindful of the context provided in the conversation history and maximally truthful.\n\n"
        "Relevant Context from Memory:\n"
        "{history}"
    ),
    HumanMessagePromptTemplate.from_template("{input}")
])

prompt = chat_prompt

conversation = ConversationChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True,
)

# --- MEMORY AUDIT BUTTON ---
st.sidebar.markdown("---")
st.sidebar.header("🔍 Memory Audit")
st.sidebar.write("See what I have learned about you.")

if st.sidebar.button("What do you know about me?"):
    audit_question = "Summarize everything you know about me: my background, my education, my projects, and my preferences. Only use factual information stored in your memory database. Do not make anything up."
    
    # 1. Add user question to UI
    st.session_state.messages.append({"role": "user", "content": "What do you know about me?"})
    
    # 2. Generate the memory summary
    with st.spinner("Scanning vector database..."):
        summary_response = conversation.predict(input=audit_question)
        
    # 3. Add AI response to UI
    st.session_state.messages.append({"role": "assistant", "content": summary_response})
    
    # Force the UI to refresh so the messages appear instantly
    st.rerun()
    
# --- 4. CHAT INTERFACE ---

# Initialize chat history for the UI session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages in the chat window
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if user_input := st.chat_input("What is on your mind?"):
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = conversation.predict(input=user_input)
            st.markdown(response)
    
    # 3. Save Response to UI History
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # 4. Save Interaction to Long-Term Memory (ChromaDB)
    # Note: ConversationChain does this automatically for the current session, 
    # but explicitly saving context ensures it persists to disk immediately.
    memory.save_context({"input": user_input}, {"output": response})