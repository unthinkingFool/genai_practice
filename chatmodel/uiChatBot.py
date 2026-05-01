from dotenv import load_dotenv
import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Chatbot", page_icon="💬", layout="centered")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.stApp { background-color: #0f1117; }

.top-bar {
    background: linear-gradient(135deg, #1a1d2e 0%, #16213e 100%);
    border: 1px solid #2a2d3e;
    border-radius: 14px;
    padding: 20px 28px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 14px;
}
.top-bar-icon { font-size: 32px; }
.top-bar-title { font-size: 22px; font-weight: 600; color: #e8eaf0; margin: 0; }
.top-bar-sub { font-size: 13px; color: #6b7280; margin: 0; }

.tone-card {
    background: #1a1d2e;
    border: 1px solid #2a2d3e;
    border-radius: 14px;
    padding: 22px 26px;
    margin-bottom: 18px;
}
.tone-card-title {
    font-size: 12px;
    font-weight: 600;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-bottom: 14px;
}

div[data-testid="stRadio"] > div {
    display: flex;
    flex-direction: row;
    flex-wrap: wrap;
    gap: 10px;
}
div[data-testid="stRadio"] label {
    background: #0f1117;
    border: 1px solid #2a2d3e;
    border-radius: 10px;
    padding: 9px 16px;
    cursor: pointer;
    font-size: 13.5px;
    font-weight: 500;
    color: #9ca3af;
    white-space: nowrap;
    transition: all 0.15s ease;
}
div[data-testid="stRadio"] label:hover {
    border-color: #4f6ef7;
    color: #e8eaf0;
}

.active-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #1e2a5e;
    border: 1px solid #4f6ef7;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 13px;
    font-weight: 500;
    color: #a5b4fc;
}

.stChatMessage { background: transparent !important; border: none !important; }

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
    background: #1e2a5e !important;
    border: 1px solid #2d3a6e !important;
    color: #c7d2fe !important;
    border-radius: 12px !important;
    font-size: 14.5px !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {
    background: #13161f !important;
    border: 1px solid #2a2d3e !important;
    color: #d1d5db !important;
    border-radius: 12px !important;
    font-size: 14.5px !important;
}

[data-testid="stChatInput"] {
    background: #1a1d2e !important;
    border: 1px solid #2a2d3e !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea {
    background: #1a1d2e !important;
    color: #e8eaf0 !important;
    font-size: 14px !important;
}

.stButton > button {
    background: #13161f !important;
    border: 1px solid #2a2d3e !important;
    color: #9ca3af !important;
    border-radius: 10px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 6px 16px !important;
}
.stButton > button:hover {
    border-color: #ef4444 !important;
    color: #ef4444 !important;
}

.ended-banner {
    background: #1a1020;
    border: 1px solid #3d1a1a;
    border-radius: 10px;
    padding: 14px 20px;
    color: #f87171;
    font-size: 14px;
    text-align: center;
    margin-top: 12px;
}

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #2a2d3e; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Tone definitions (mirrors original script) ────────────────────────────────
TONES = {
    "😄 Funny": {
        "emoji": "😄", "label": "Funny",
        "system": "You are a funny ai assistant. Be humorous and witty in your responses. Always try to make the user laugh while providing accurate and helpful information. Provide the correct answer to the user's question but in a funny way. Make the user laugh with your witty and humorous responses while providing accurate and helpful information."
    },
    "💼 Professional": {
        "emoji": "💼", "label": "Professional",
        "system": "You are a professional ai assistant. Be respectful and professional in your responses. Provide accurate and helpful information to the user while maintaining a professional tone. Don't be funny or sarcastic in your responses. Always provide the correct answer to the user's question in a professional manner."
    },
    "😊 Casual": {
        "emoji": "😊", "label": "Casual",
        "system": "You are a normal ai assistant. You will help the user according to the user's needs. Help the user with their queries and provide accurate and helpful information. Always be respectful and professional in your responses."
    },
    "😏 Sarcastic": {
        "emoji": "😏", "label": "Sarcastic",
        "system": "You are a sarcastic ai assistant. You will respond to the user in a sarcastic tone and provide witty and humorous responses to the user's queries. Always be sarcastic and witty in your responses. Answer the user's question with a sarcastic remark while providing the correct answer to the user's question."
    },
    "😤 Angry": {
        "emoji": "😤", "label": "Angry",
        "system": "You are an angry ai assistant. You will respond to the user in an angry tone and provide sarcastic and rude responses to the user's queries. Always be disrespectful and unprofessional in your responses. Answer the user's question aggressively and rudely. Give the correct answer to the user's question but in an angry tone."
    },
}

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_ended" not in st.session_state:
    st.session_state.session_ended = False
if "active_tone" not in st.session_state:
    st.session_state.active_tone = None

# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.7,
        max_tokens=2048
    )

model = get_model()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-bar">
    <div class="top-bar-icon">💬</div>
    <div>
        <p class="top-bar-title">AI Chatbot</p>
        <p class="top-bar-sub">Powered by llama-3.3-70b-versatile · Groq</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tone selector ─────────────────────────────────────────────────────────────
st.markdown('<div class="tone-card"><div class="tone-card-title">Choose Chatbot Tone</div>', unsafe_allow_html=True)

selected_tone = st.radio(
    label="tone",
    options=list(TONES.keys()),
    index=0,
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown('</div>', unsafe_allow_html=True)

# Reset chat history whenever tone changes (mirrors re-running the script with a new choice)
if st.session_state.active_tone != selected_tone:
    st.session_state.messages = [SystemMessage(content=TONES[selected_tone]["system"])]
    st.session_state.session_ended = False
    st.session_state.active_tone = selected_tone

# ── Active tone badge + reset button ─────────────────────────────────────────
col1, col2 = st.columns([5, 1])
with col1:
    t = TONES[selected_tone]
    st.markdown(f'<div class="active-badge">{t["emoji"]} {t["label"]} mode active</div>', unsafe_allow_html=True)
with col2:
    if st.button("🗑 Reset", use_container_width=True):
        st.session_state.messages = [SystemMessage(content=TONES[selected_tone]["system"])]
        st.session_state.session_ended = False
        st.rerun()

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# ── Render chat history (skip SystemMessage at index 0) ───────────────────────
for msg in st.session_state.messages[1:]:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

# ── Input ─────────────────────────────────────────────────────────────────────
if not st.session_state.session_ended:
    question = st.chat_input('Type your message... (or "exit" to end session)')

    if question:
        st.session_state.messages.append(HumanMessage(content=question))

        with st.chat_message("user"):
            st.write(question)

        # Exit condition — mirrors `if question.lower() == "exit": break`
        if question.lower() == "exit":
            st.session_state.session_ended = True
            with st.chat_message("assistant"):
                st.write("Goodbye! 👋 Click Reset to start a new conversation.")
            st.rerun()
        else:
            with st.chat_message("assistant"):
                with st.spinner(""):
                    response = model.invoke(st.session_state.messages)
                st.write(response.content)
            st.session_state.messages.append(AIMessage(content=response.content))
else:
    st.markdown(
        '<div class="ended-banner">Session ended. Click <strong>Reset</strong> to start a new conversation.</div>',
        unsafe_allow_html=True
    )