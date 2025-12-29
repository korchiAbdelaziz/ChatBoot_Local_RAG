import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
from PIL import Image
import os
import requests
import json
import numpy as np
from rag_engine import build_index, search, DOCS_DIR

# --- CORRECTIF TECHNIQUE ---
class CompatibleDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(**kwargs)

# ---------- CONFIG PAGE ----------
st.set_page_config(page_title="RAG Neural Chat", page_icon="🤖", layout="centered")

# ---------- STYLE CYBER-CHAT & ANIMATIONS ----------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Rajdhani:wght@500;700&display=swap');

    /* BACKGROUND IMMERSIF */
    .stApp {
        background: linear-gradient(rgba(0, 0, 0, 0.9), rgba(0, 0, 0, 0.9)), 
                    url("https://r.jina.ai/i/9e063878772348508e6473c180862086");
        background-attachment: fixed;
        background-size: cover;
        font-family: 'Rajdhani', sans-serif;
        color: #e0e0e0;
    }

    /* TITRE NÉON */
    .cyber-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 2rem;
        text-align: center;
        background: linear-gradient(90deg, #00fbff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 15px rgba(0, 251, 255, 0.4);
    }

    /* CONTENEUR DE CHAT ANIMÉ */
    [data-testid="stChatMessage"] {
        background: rgba(10, 20, 30, 0.7) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 251, 255, 0.1);
        border-radius: 15px !important;
        margin-bottom: 10px;
        animation: slideUp 0.5s ease-out;
    }

    /* BUBBLE USER VS ASSISTANT */
    [data-testid="stChatMessage"]:nth-child(even) {
        border-left: 4px solid #00fbff !important;
    }

    /* ANIMATION */
    @keyframes slideUp {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }

    /* SIDEBAR GLASSMORPISM */
    [data-testid="stSidebar"] {
        background-color: rgba(5, 10, 15, 0.95) !important;
        border-right: 1px solid rgba(0, 251, 255, 0.2);
    }

    /* INPUT CHAT FIXÉ EN BAS (RESPONSIVE) */
    .stChatInputContainer {
        padding-bottom: 20px;
    }

    /* CACHER ÉLÉMENTS STREAMLIT */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ------------------------------
#   LOGIQUE INITIALISATION
# ------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ------------------------------
#   SIDEBAR PARAMÈTRES
# ------------------------------
with st.sidebar:
    st.markdown("<h2 style='color:#00fbff; font-family:Orbitron; font-size:1.2rem;'>⚙️ PARAMÈTRES</h2>", unsafe_allow_html=True)
    model_name = st.text_input("Ollama Model", value="llama3.2:latest")
    
    st.divider()
    st.markdown("<h2 style='color:#00fbff; font-family:Orbitron; font-size:1.2rem;'>📄 DOCUMENTS</h2>", unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["pdf", "txt", "md"])

    if uploaded:
        os.makedirs(DOCS_DIR, exist_ok=True)
        with open(os.path.join(DOCS_DIR, uploaded.name), "wb") as f:
            f.write(uploaded.getbuffer())
        st.success("Fichier synchronisé")
        if st.button("Indexer le cerveau"):
            with st.spinner("Indexation..."):
                st.info(build_index())

# ------------------------------
#   UI PRINCIPALE
# ------------------------------
st.markdown('<h1 class="cyber-title">RAG NEURAL CHAT</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center; opacity:0.6; font-size:0.8rem;'>OLLAMA + LOCAL KNOWLEDGE BASE</p>", unsafe_allow_html=True)

# Affichage des messages
for turn in st.session_state.history:
    role = turn["role"]
    if role != "system":
        with st.chat_message(role):
            st.write(turn["content"])

# ------------------------------
#   FONCTIONS APPEL
# ------------------------------
def call_ollama(prompt):
    url = "http://localhost:11434/api/generate"
    payload = {"model": model_name, "prompt": prompt, "stream": False}
    try:
        resp = requests.post(url, json=payload)
        return json.loads(resp.text).get("response", "Erreur de réponse")
    except Exception as e:
        return f"Erreur de connexion: {e}"

# ------------------------------
#   INPUT & RAG
# ------------------------------
if user_msg := st.chat_input("Interroger la base de données..."):
    
    # Affichage immédiat du message user
    st.session_state.history.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.write(user_msg)

    # Recherche RAG
    with st.spinner("Accès aux archives..."):
        context = search(user_msg, k=3)

    # Construction du Prompt
    final_prompt = f"Context:\n{context}\n\nQuestion: {user_msg}\nAnswer:"

    # Réponse Assistant
    with st.chat_message("assistant"):
        with st.spinner("Calcul des neurones..."):
            reply = call_ollama(final_prompt)
            st.write(reply)
            st.session_state.history.append({"role": "assistant", "content": reply})

    st.rerun()