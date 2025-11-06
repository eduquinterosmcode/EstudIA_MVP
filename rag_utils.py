# Archivo: rag_utils.py
# (El "Motor" RAG - Versión 1.1 Corregida)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter  # <- ¡LA IMPORTACIÓN CLAVE!
import streamlit as st


# Usamos st.cache_resource para "guardar" el RAG y no reconstruirlo cada vez
# Esto es CRUCIAL para que el demo sea rápido.
@st.cache_resource
def build_rag_chain(transcript_text, api_key):
    """
    Toma una transcripción de texto y construye una cadena (chain) de RAG
    que puede responder preguntas sobre ese texto.
    """
    st.info("Construyendo motor RAG por primera vez... (Esto se guardará en caché)")
    
    # 1. Dividir el Texto (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    docs = text_splitter.create_documents([transcript_text])
    
    if not docs:
        st.error("Error: El texto no pudo ser dividido. ¿El audio estaba vacío?")
        return None

    # 2. Crear Embeddings (Vectorizar)
    embeddings = OpenAIEmbeddings(api_key=api_key)

    # 3. Crear VectorStore (Base de Datos FAISS en memoria)
    try:
        vector_store = FAISS.from_documents(docs, embeddings)
    except Exception as e:
        st.error(f"Error creando la base de datos vectorial (FAISS): {e}")
        st.error("Asegúrate de que 'faiss-cpu' se instaló correctamente.")
        return None

    # 4. Crear el Retriever
    retriever = vector_store.as_retriever()

    # 5. Crear el Prompt
    prompt = ChatPromptTemplate.from_template("""
    Eres "EstudIA", un asistente experto analizando una presentación.
    Tu misión es responder las preguntas del usuario basándote ÚNICA y EXCLUSIVAMENTE 
    en el siguiente contexto (la transcripción del pitch).
    Si la respuesta no está en el contexto, di amablemente: 
    "Lo siento, no tengo información sobre eso en la transcripción de la presentación."

    Contexto (Transcripción):
    {context}

    Pregunta del Usuario:
    {input}
    """)

    # 6. Definir el Modelo
    llm = ChatOpenAI(
        model_name="gpt-4-turbo", 
        temperature=0.3, 
        api_key=api_key
    )

    # 7. Crear la Cadena usando el nuevo sistema LCEL (¡Versión Corregida!)
    # Esta cadena ahora sabe cómo "extraer" el texto del diccionario de entrada
    chain = (
        {
            "context": itemgetter("input") | retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
            "input": itemgetter("input")
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    st.success("¡Motor RAG construido y listo!")
    return chain