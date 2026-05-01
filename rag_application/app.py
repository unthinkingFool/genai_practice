import streamlit as st
import os
import tempfile
import threading
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocChat · RAG Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* ── Background ── */
.stApp {
    background: #0d0f14;
    color: #e8e4dc;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #111318;
    border-right: 1px solid #1f2330;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: #9fa3b0 !important;
}

/* ── Brand header ── */
.brand-header {
    padding: 2rem 0 1.5rem;
    border-bottom: 1px solid #1f2330;
    margin-bottom: 1.5rem;
}
.brand-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.7rem;
    color: #e8e4dc;
    letter-spacing: -0.02em;
    margin: 0;
    line-height: 1.2;
}
.brand-sub {
    font-size: 0.78rem;
    color: #5a5f72;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 0.3rem;
}

/* ── Status pill ── */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.06em;
    margin-top: 0.8rem;
}
.status-ready { background: #0d2b1f; color: #3ecf8e; border: 1px solid #1a4a35; }
.status-idle  { background: #1a1c23; color: #5a5f72; border: 1px solid #252836; }

/* ── File slots ── */
.file-counter {
    font-size: 0.72rem;
    color: #5a5f72;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

/* ── Uploaded file list ── */
.doc-list-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 10px;
    border-radius: 6px;
    background: #161820;
    border: 1px solid #1f2330;
    margin-bottom: 6px;
    font-size: 0.82rem;
    color: #b0b5c4;
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
}
.doc-icon { font-size: 0.9rem; flex-shrink: 0; }

/* ── Process button ── */
.stButton > button {
    width: 100%;
    background: #c9a96e;
    color: #0d0f14;
    border: none;
    padding: 0.6rem 1rem;
    border-radius: 6px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.04em;
    cursor: pointer;
    transition: background 0.2s, transform 0.1s;
}
.stButton > button:hover { background: #e0be8a; }
.stButton > button:active { transform: scale(0.98); }
.stButton > button:disabled {
    background: #252836;
    color: #3d4157;
    cursor: not-allowed;
}

/* ── Main area header ── */
.main-header {
    padding: 2.5rem 0 1rem;
}
.main-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    color: #e8e4dc;
    letter-spacing: -0.03em;
    line-height: 1.1;
}
.main-title em { color: #c9a96e; font-style: italic; }
.main-desc {
    color: #5a5f72;
    font-size: 0.9rem;
    margin-top: 0.5rem;
    max-width: 520px;
    line-height: 1.6;
}

/* ── Chat messages ── */
.chat-wrapper {
    max-width: 760px;
}

[data-testid="stChatMessage"] {
    border-radius: 10px;
    margin-bottom: 0.5rem;
    border: 1px solid #1f2330;
    background: #111318 !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: #111318;
    border: 1px solid #1f2330;
    border-radius: 8px;
    color: #e8e4dc;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #c9a96e;
    box-shadow: 0 0 0 2px rgba(201,169,110,0.12);
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: #3d4157;
}
.empty-state-icon { font-size: 3rem; margin-bottom: 1rem; opacity: 0.4; }
.empty-state-text { font-size: 0.9rem; line-height: 1.6; }

/* ── Divider ── */
hr { border-color: #1f2330 !important; margin: 1.2rem 0 !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #c9a96e !important; }

/* ── Info / warning boxes ── */
.stAlert { border-radius: 8px; border: 1px solid #1f2330; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "doc_names" not in st.session_state:
    st.session_state.doc_names = []


# ── Helpers ──────────────────────────────────────────────────────────────────
MAX_FILES = 5

@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def get_model():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.7,
        max_tokens=2048,
    )

def _process_single_file(uf, result_list, errors_list, lock):
    """
    Runs in its own thread. Loads one file, splits it into chunks,
    and appends those chunks to the shared result_list.
    """
    suffix = os.path.splitext(uf.name)[1].lower()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uf.read())
            tmp_path = tmp.name

        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path, encoding="utf-8")

        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        with lock:
            result_list.extend(chunks)

    except Exception as e:
        with lock:
            errors_list.append(f"{uf.name}: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def build_retriever(uploaded_files):
    """
    Spawns one thread per uploaded file to load & chunk in parallel.
    All chunks are merged, then embedded and stored in a single Chroma vectorstore.
    """
    all_chunks = []
    errors = []
    lock = threading.Lock()

    threads = []
    for uf in uploaded_files:
        t = threading.Thread(
            target=_process_single_file,
            args=(uf, all_chunks, errors, lock),
            name=f"loader-{uf.name}",
            daemon=True,
        )
        threads.append(t)

    # Start all threads (one per file)
    for t in threads:
        t.start()

    # Wait for every thread to finish before embedding
    for t in threads:
        t.join()

    if errors:
        raise RuntimeError("Some files failed to process:\n" + "\n".join(errors))

    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
    )
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5},
    )

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful AI assistant.\n\n"
     "Use ONLY the provided context to answer the question.\n\n"
     "If the answer is not present in the context, "
     'say: "I could not find the answer in the document."'),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
])


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="brand-header">
        <p class="brand-title">DocChat</p>
        <p class="brand-sub">RAG · Document Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

    # Status pill
    if st.session_state.retriever:
        n = len(st.session_state.doc_names)
        st.markdown(
            f'<div class="status-pill status-ready">● {n} document{"s" if n>1 else ""} loaded</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="status-pill status-idle">○ No documents loaded</div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # File counter
    st.markdown('<div class="file-counter">Upload documents (max 5)</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        label="Upload documents",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    # Enforce max
    if uploaded_files and len(uploaded_files) > MAX_FILES:
        st.warning(f"Only the first {MAX_FILES} files will be used.")
        uploaded_files = uploaded_files[:MAX_FILES]

    # Show uploaded file list
    if uploaded_files:
        st.markdown("<br>", unsafe_allow_html=True)
        for uf in uploaded_files:
            icon = "📄" if uf.name.endswith(".pdf") else "📝"
            st.markdown(
                f'<div class="doc-list-item"><span class="doc-icon">{icon}</span>{uf.name}</div>',
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    process_btn = st.button(
        "⚡ Process Documents",
        disabled=not uploaded_files,
        use_container_width=True,
    )

    if process_btn and uploaded_files:
        with st.spinner("Embedding documents…"):
            try:
                st.session_state.retriever = build_retriever(uploaded_files)
                st.session_state.doc_names = [f.name for f in uploaded_files]
                st.session_state.messages = []
                st.success("Ready! Start asking questions.")
                st.rerun()
            except Exception as e:
                st.error(f"Error processing documents: {e}")

    st.markdown("<hr>", unsafe_allow_html=True)

    if st.session_state.retriever:
        if st.button("🗑 Clear Session", use_container_width=True):
            st.session_state.retriever = None
            st.session_state.doc_names = []
            st.session_state.messages = []
            st.rerun()

    st.markdown("""
    <br>
    <div style="color:#3d4157;font-size:0.72rem;line-height:1.7;">
        Supports <strong style="color:#4d5270">PDF</strong> and 
        <strong style="color:#4d5270">TXT</strong> files.<br>
        Powered by <strong style="color:#4d5270">Groq · LLaMA 3.3</strong><br>
        Embeddings via <strong style="color:#4d5270">MiniLM-L6-v2</strong>
    </div>
    """, unsafe_allow_html=True)


# ── Main area ─────────────────────────────────────────────────────────────────
col_main, col_pad = st.columns([3, 1])

with col_main:
    st.markdown("""
    <div class="main-header">
        <div class="main-title">Ask your <em>documents</em></div>
        <p class="main-desc">
            Upload PDFs or text files, process them once, then ask anything.
            Answers are grounded strictly in your documents.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Chat history
    if not st.session_state.retriever:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">📂</div>
            <div class="empty-state-text">
                Upload up to 5 documents in the sidebar<br>and click <strong>Process Documents</strong> to begin.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # Chat input
        if question := st.chat_input("Ask a question about your documents…"):
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        model = get_model()
                        docs = st.session_state.retriever.invoke(question)
                        context = "\n\n".join([d.page_content for d in docs])
                        final_prompt = PROMPT.format(context=context, question=question)
                        response = model.invoke(final_prompt)
                        answer = response.content
                    except Exception as e:
                        answer = f"⚠️ Error generating answer: {e}"

                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})