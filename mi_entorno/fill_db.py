import os
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from googletrans import Translator
from langdetect import detect
from tqdm import tqdm  # Para mostrar progreso
import logging  # Para manejo de errores

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ------------------------------
# Loader para archivos SQL y otros formatos
# ------------------------------

class DocumentLoader:
    def __init__(self, path):
        self.path = path
        
    def load_sql(self):
        """Carga archivos SQL/SQLite y extrae declaraciones relevantes"""
        documentos = []
        sql_extensions = ['.sql', '.sqlite']
        
        for root, _, files in os.walk(self.path):
            for file in files:
                if any(file.endswith(ext) for ext in sql_extensions):
                    try:
                        full_path = os.path.join(root, file)
                        with open(full_path, 'r', encoding='utf-8') as f:
                            sql_content = f.read()
                            
                            # Separar sentencias por punto y coma
                            statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
                            
                            for stmt in statements:
                                if any(stmt.upper().startswith(prefix) for prefix in ["CREATE", "ALTER", "INSERT", "SELECT"]):
                                    documentos.append(
                                        Document(
                                            page_content=stmt,
                                            metadata={
                                                "source": full_path,
                                                "type": "sql",
                                                "file_name": file
                                            }
                                        )
                                    )
                    except Exception as e:
                        logger.error(f"Error procesando archivo {file}: {str(e)}")
        
        return documentos
    
    def load_all(self):
        """Carga todos los documentos soportados"""
        pdf_loader = PyPDFDirectoryLoader(self.path)
        
        try:
            pdf_docs = pdf_loader.load()
            logger.info(f"Cargados {len(pdf_docs)} documentos PDF")
        except Exception as e:
            pdf_docs = []
            logger.error(f"Error cargando PDFs: {str(e)}")
        
        try:
            sql_docs = self.load_sql()
            logger.info(f"Cargados {len(sql_docs)} fragmentos SQL")
        except Exception as e:
            sql_docs = []
            logger.error(f"Error cargando SQLs: {str(e)}")
            
        return pdf_docs + sql_docs

# ------------------------------
# Función para traducir documentos
# ------------------------------

def translate_documents(documents, target_lang='es', batch_size=5):
    """Traduce documentos al idioma objetivo si es necesario"""
    translator = Translator()
    translated_docs = []
    
    for i in tqdm(range(0, len(documents), batch_size), desc="Traduciendo documentos"):
        batch = documents[i:i+batch_size]
        
        for doc in batch:
            try:
                # Comprobar si el texto es suficientemente largo para detectar el idioma
                if len(doc.page_content.strip()) > 10:
                    detected_language = detect(doc.page_content)
                    
                    # Traducir solo si no está en el idioma objetivo
                    if detected_language != target_lang:
                        translated_content = translator.translate(
                            doc.page_content, 
                            src=detected_language, 
                            dest=target_lang
                        ).text
                        
                        # Actualizar metadata para rastrear traducción
                        updated_metadata = doc.metadata.copy()
                        updated_metadata["original_language"] = detected_language
                        updated_metadata["translated"] = True
                    else:
                        translated_content = doc.page_content
                        updated_metadata = doc.metadata.copy()
                        updated_metadata["translated"] = False
                else:
                    # Para textos muy cortos, mantener original
                    translated_content = doc.page_content
                    updated_metadata = doc.metadata.copy()
                    updated_metadata["translated"] = False
                
                translated_docs.append(
                    Document(
                        page_content=translated_content,
                        metadata=updated_metadata
                    )
                )
            except Exception as e:
                logger.warning(f"Error traduciendo documento: {str(e)}")
                # Mantener el documento original en caso de error
                translated_docs.append(doc)
    
    return translated_docs

# ------------------------------
# Principal
# ------------------------------

def main():
    DATA_PATH = r"data"
    CHROMA_PATH = r"chroma_db"
    
    # Asegurar que los directorios existen
    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(CHROMA_PATH, exist_ok=True)
    
    # Inicializar cliente Chroma
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(name="nba_regulation")
    
    # Cargar documentos
    logger.info("Iniciando carga de documentos...")
    document_loader = DocumentLoader(DATA_PATH)
    raw_documents = document_loader.load_all()
    logger.info(f"Cargados {len(raw_documents)} documentos en total")
    
    # Traducir documentos
    logger.info("Traduciendo documentos...")
    translated_documents = translate_documents(raw_documents)
    logger.info(f"Procesados {len(translated_documents)} documentos para traducción")
    
    # Split del texto en chunks
    logger.info("Dividiendo documentos en chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = text_splitter.split_documents(translated_documents)
    logger.info(f"Creados {len(chunks)} chunks para indexar")
    
    # Preparar datos para Chroma
    documents = []
    metadata = []
    ids = []
    
    for i, chunk in enumerate(chunks):
        documents.append(chunk.page_content)
        ids.append(f"chunk_{i}")
        metadata.append(chunk.metadata)
    
    # Indexar en Chroma
    logger.info("Indexando en Chroma...")
    collection.upsert(
        documents=documents,
        metadatas=metadata,
        ids=ids
    )
    logger.info(f"Indexación completada - {len(documents)} chunks indexados")

if __name__ == "__main__":
    main()