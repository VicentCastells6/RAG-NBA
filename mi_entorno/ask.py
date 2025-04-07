import streamlit as st
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "chroma_db"

# Setup LLM y vector store
llm = OllamaLLM(model="llama3")
embedding = OllamaEmbeddings(model="llama3")
vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding)
# Aumentar k para recuperar más contexto relevante
retriever = vectordb.as_retriever(search_kwargs={"k": 8})

# Prompt con contexto y directrices fijas
prompt_template = """
Eres un asistente experto en la NBA. Responde únicamente usando el contexto proporcionado. Si no encuentras información suficiente, dilo claramente.

Reglas importantes:
- Responde solo si tienes información en el contexto.
- Si la pregunta no tiene que ver con la NBA o el baloncesto profesional, responde: "Lo siento, solo puedo responder preguntas relacionadas con la NBA."
- Si no tienes información suficiente, responde: "No tengo información suficiente para responder esa pregunta."

--------------------
CONTEXTO:
{context}

PREGUNTA:
{question}

RESPUESTA:
"""



prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True 
)

st.set_page_config(page_title="Asistente NBA", page_icon="🏀")

# Mostrar logo y título alineados
logo_col, title_col = st.columns([1, 5])
with logo_col:
    st.image("mi_entorno\src\logo.png", width=60)

with title_col:
    st.title("🏀 Asistente NBA")
    st.markdown("Pregúntame lo que quieras sobre el reglamento, jugadores, equipos e historia de la NBA.")

# Estado para historial de chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.markdown("""
    <style>
    .chat-container {
        height: 900px;
        width: 100%;
        overflow-y: auto;
        background-color: #0E1117;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #0E1117;
        color: white;
        margin-bottom: 15px;
    }
    .user-msg {
        background-color: #004466;
        padding: 10px 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        max-width: 80%;
        margin-left: 10px;
        margin-right: auto;
        text-align: left;
    }
    .bot-msg {
        background-color: #222;
        padding: 10px 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        max-width: 80%;
        margin-right: 10px;
        margin-left: auto;
        text-align: right;
    }
    .stButton > button {
        height: 40px;
        width: 100%;
    }
    .input-container {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    </style>
""", unsafe_allow_html=True)

chat_container = st.container()

# st.empty() se usa para actualizar dinámicamente
with chat_container:
    chat_placeholder = st.empty()
    

    chat_html = '<div class="chat-container">'
    for q, a in st.session_state.chat_history:
        chat_html += f'<div class="user-msg">{q}</div>'
        chat_html += f'<div class="bot-msg">{a}</div>'
    chat_html += '</div>'
    

    chat_placeholder.markdown(chat_html, unsafe_allow_html=True)

# Función para procesar la consulta
def process_query():
    query = st.session_state.user_input
    if query.strip():
        with st.spinner("Buscando información..."):
            try:
                response = qa_chain.invoke({"query": query})
                answer = response["result"]
                
                st.session_state.chat_history.append((query, answer))
                
                st.session_state.user_input = ""
                
                
            except Exception as e:
                st.error(f"Ocurrió un error: {str(e)}")

# Inicializar el estado del input si no existe
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

st.container()
cols = st.columns([5, 1])

with cols[0]:
    st.text_input(
        key="user_input",
        placeholder="Ejemplo: ¿Cuánto dura un partido de NBA?",
        on_change=process_query,
        label="Escribe tu pregunta aquí:",
        label_visibility="collapsed",
        max_chars=200,
        args=(),
        kwargs={},
    )

with cols[1]:
    st.button(
        "Preguntar", 
        key="submit",
        on_click=process_query,
        args=(),
        kwargs={},
    )