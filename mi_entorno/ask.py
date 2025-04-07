from flask import Flask, request, jsonify
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

# Configuración de Chroma
CHROMA_PATH = "chroma_db"

# Setup LLM y vector store
llm = OllamaLLM(model="llama3")
embedding = OllamaEmbeddings(model="llama3")
vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding)
retriever = vectordb.as_retriever(search_kwargs={"k": 8})

# Template para respuestas contextuales
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

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "").strip()
    
    if not question:
        return jsonify({"error": "Pregunta vacía"}), 400

    try:
        response = qa_chain.invoke({"query": question})
        return jsonify({
            "question": question,
            "answer": response["result"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "message": "NBA Chatbot API funcionando 😎"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
