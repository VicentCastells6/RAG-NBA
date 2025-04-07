# 🏀 Asistente NBA con Llama 3 y Chroma

¡Qué lo qué, mi rey! Este proyecto es un **asistente inteligente de la NBA** hecho con **Streamlit**, **LangChain**, **ChromaDB** y el motorcito de IA **Llama3**. Ideal pa’ cuando tú quieras preguntar vainas como “¿cuánto mide LeBron sin tenis?” o “¿qué carajo es un pick and roll?”.

---

## 📦 ¿Qué hace esta vaina?

- Te responde preguntas sobre la NBA con información sacada de documentos PDF y archivos SQL.
- Te traduce documentos si vienen en otro idioma (pa' que no te joda el inglés).
- Usa embeddings y recuperación de contexto pa’ que no se invente vainas.
- Tiene una interfaz heavy pa’ usar en el navegador.

---

## 🚀 ¿Cómo se monta esta vaina?

1. **Clona el repo:**

   ```bash
   git clone https://tu-repo.com/nba-asistente.git
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

