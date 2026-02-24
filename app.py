import streamlit as st
import os
from dotenv import load_dotenv
import tempfile
import fitz  # PyMuPDF
from rapidocr_onnxruntime import RapidOCR

# LangChain Imports
from langchain_community.chat_models import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

load_dotenv()

# --- CONFIGURATION ---
PAGE_TITLE = "My Offline AI Agent"
MODEL_NAME = "llama3"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
PERSIST_DIR = "./ai_memory_db"

st.set_page_config(page_title=PAGE_TITLE, layout="wide")
st.title(PAGE_TITLE)

# --- 1. INITIALIZE RESOURCES (Dual Collections) ---
@st.cache_resource
def get_vector_stores():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Collection A: The Vault (Strictly for PDFs, TXT, Logs)
    doc_db = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name="document_vault"
    )
    
    # Collection B: Organic Memory (Strictly for Chat History)
    chat_db = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name="episodic_memory"
    )
    return doc_db, chat_db

doc_db, chat_db = get_vector_stores()

# --- 2. SIDEBAR: KNOWLEDGE INGESTION ---
st.sidebar.header("🧠 Knowledge Base")
st.sidebar.write("Upload documents (PDF, TXT, LOG) to give your AI long-term memory.")

uploaded_file = st.sidebar.file_uploader("Choose a file...", type=['pdf', 'txt', 'log'])

def get_text_from_scanned_pdf(pdf_path):
    ocr = RapidOCR()
    doc = fitz.open(pdf_path)
    full_text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img_bytes = pix.tobytes("png")
        result, _ = ocr(img_bytes)
        if result:
            for line in result:
                full_text += line[1] + "\n"
    return full_text

if uploaded_file is not None:
    with st.spinner("Processing... (If scanned, this may take a moment)"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            documents = []
            if uploaded_file.name.endswith('.pdf'):
                loader = PyPDFLoader(tmp_file_path)
                documents = loader.load()
                
                if not documents or len(documents[0].page_content.strip()) < 10:
                    st.sidebar.warning("Scanned document detected! Switching to AI OCR (Slower)...")
                    scanned_text = get_text_from_scanned_pdf(tmp_file_path)
                    documents = [Document(page_content=scanned_text, metadata={"source": uploaded_file.name})]
            else:
                loader = TextLoader(tmp_file_path, encoding='utf-8')
                documents = loader.load()

            if documents and len(documents[0].page_content) > 0:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_documents(documents)
                # Save to Vault!
                doc_db.add_documents(chunks)
                st.sidebar.success(f"✅ Memorized {len(chunks)} chunks from {uploaded_file.name}!")
            else:
                st.sidebar.error("Could not read any text from this file.")
        
        except Exception as e:
            st.sidebar.error(f"Error processing file: {e}")
        
        finally:
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)


# --- 3. SET UP THE BRAIN ---
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Model Settings")
llm_choice = st.sidebar.radio(
    "Select your AI Brain:", 
    ["Local Offline (Llama 3)", "Cloud API (Gemini 3.1 Pro)"]
)

if llm_choice == "Local Offline (Llama 3)":
    try:
        llm = ChatOllama(model="llama3")
    except Exception as e:
        st.error("Make sure Ollama is running in your terminal!")
        st.stop()
else:
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview")

# Unified Prompt for Dual-Memory
prompt_template = ChatPromptTemplate.from_template("""You are a highly capable personal AI assistant.
Use the following context to answer the user's question. Prioritize factual documents over chat history if there is a conflict.

[FACTUAL DOCUMENT KNOWLEDGE]
{doc_context}

[PAST CONVERSATIONAL MEMORY]
{chat_context}

Current Conversation:
Human: {user_input}
AI:""")

# --- MEMORY AUDIT BUTTON ---
st.sidebar.markdown("---")
st.sidebar.header("🔍 Memory Audit")
st.sidebar.write("See what I have learned about you.")

if st.sidebar.button("What do you know about me?"):
    audit_question = "Summarize everything you know about me: my background, my education, my projects, and my preferences. Only use factual information stored in your memory databases. Do not make anything up."
    
    st.session_state.messages.append({"role": "user", "content": "What do you know about me?"})
    
    with st.spinner("Scanning both vector databases..."):
        # Manually query both databases
        docs = doc_db.similarity_search(audit_question, k=3)
        chats = chat_db.similarity_search(audit_question, k=3)
        
        doc_context = "\n".join([d.page_content for d in docs])
        chat_context = "\n".join([c.page_content for c in chats])
        
        # Invoke the LLM
        chain = prompt_template | llm
        response_message = chain.invoke({
            "doc_context": doc_context,
            "chat_context": chat_context,
            "user_input": audit_question
        })
        summary_response = response_message.content
        
    st.session_state.messages.append({"role": "assistant", "content": summary_response})
    st.rerun()

# --- 4. CHAT INTERFACE & CUSTOM EXECUTION LOOP ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("What is on your mind?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching files and past memories..."):
            
            # 1. Retrieve from Collection A (Documents)
            docs = doc_db.similarity_search(user_input, k=3)
            doc_context = "\n".join([d.page_content for d in docs])
            
            # 2. Retrieve from Collection B (Past Chats)
            chats = chat_db.similarity_search(user_input, k=3)
            chat_context = "\n".join([c.page_content for c in chats])
            
            # 3. Build the prompt and get the LLM response
            chain = prompt_template | llm
            response_message = chain.invoke({
                "doc_context": doc_context,
                "chat_context": chat_context,
                "user_input": user_input
            })
            
            response_text = response_message.content
            st.markdown(response_text)
    
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    
    # 4. ORGANIC LEARNING: Save the interaction ONLY to the Chat Database
    memory_string = f"Human said: {user_input}\nAI replied: {response_text}"
    chat_db.add_texts([memory_string])