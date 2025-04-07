# 🏀 Asistente NBA con Llama 3 y Chroma

Este proyecto es un **asistente inteligente de la NBA** hecho con **Streamlit**, **LangChain**, **ChromaDB** y el motor de IA **Llama3**. Es una prueba de como funciona la IA y el uso de RAG para que el chatbot se focalice en los conocimientos aportados

---

## 📦 ¿Qué hace?

- Te responde preguntas sobre la NBA con información sacada de documentos PDF y archivos SQL.
- Te traduce documentos si vienen en inglés, para poder hacer las consultas en español sin problemas.
- Usa embeddings y recuperación de contexto para que no se invente vainas.
- Tiene una interfaz heavy pa’ usar en el navegador.

---

## 🚀 ¿Cómo se monta?

1. **Clona el repo:**

   ```bash
   https://github.com/VicentCastells6/RAG-NBA
   
   cd nba-asistente

## Instala los paquetes:

Usa tu venv favorito o el conda, como tú quieras. Luego:

   ```bash
    pip install -r requirements.txt
   ```
  Pon tus datos en la carpeta data/
## PDF con reglas de NBA

Bases de datos en .sql o .sqlite si tienes
[Base de datos de prueba NBA](https://www.kaggle.com/datasets/wyattowalsh/basketball?resource=download)
Llena la base de vectores:
```bash
python fill_db.py
```
Esto parte los documentos en pedacitos, los traduce si es necesario, y los mete en la base de datos chroma_db.

## Lanza la app:
```bash
streamlit run ask.py

