# Archivo: app.py (Versión 5.0 - MVP Final Modular)

import streamlit as st
from openai import OpenAI
import io 

# --- Importamos NUESTRO propio motor de RAG ---
from rag_utils import build_rag_chain

# --- 0. INICIALIZACIÓN DEL CLIENTE OPENAI ---
try:
    API_KEY = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(api_key=API_KEY)
except Exception as e:
    st.error(f"Error al inicializar cliente OpenAI: {e}")
    st.stop()

# --- 1. CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="EstudIA MVP",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
)



# --- 2. TÍTULO Y DESCRIPCIÓN ---
st.title("🤖 Bienvenido al MVP de EstudIA")
st.markdown("Súbanse al barco. ¡Vamos a transformar la educación!")

# --- 3. SEPARACIÓN DE PÁGINAS ---
tab1, tab2 = st.tabs([
    "🎙️ Función 1: Transcriptor y Resumidor", 
    "🧑‍🏫 Funciones 2 y 3: Chat Interactivo"
])

# --- PESTAÑA 1: RESUMIDOR DE CLASES (Ahora con RAG) ---
with tab1:
    st.header("🎙️ Función 1: Transcriptor y Resumidor")
    st.write("Sube el audio de una clase y obtén tu resumen. Luego puedes activa un chat para preguntar sobre él.")
    
    uploaded_audio = st.file_uploader(
        "Sube tu archivo de audio (mp3, m4a, wav):", 
        type=["mp3", "m4a", "wav"],
        key="audio_uploader" 
    )
    
    if uploaded_audio:
        st.audio(uploaded_audio)
        
        if st.button("Transcribir y Resumir Audio"):
            with st.spinner("Procesando audio... (Esto puede tardar un momento) ⏳"):
                try:
                    # 1. TRANSCRIPCIÓN CON WHISPER
                    st.subheader("1. Transcripción de la Clase (Whisper)")
                    audio_bytes = io.BytesIO(uploaded_audio.read())
                    audio_bytes.name = uploaded_audio.name
                    transcript_response = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_bytes,
                        response_format="text"
                    )
                    transcript_text = transcript_response
                    st.success("¡Audio transcrito!")
                    st.text_area("Texto de la clase:", transcript_text, height=150)

                    # --- INICIO DEL "HACK" DE RAG ---
                    st.session_state["pitch_transcrito"] = transcript_text
                    st.success("¡Transcripción guardada! Ve a la Pestaña 2 y selecciona 'Modo: Chat con tu Pitch' para preguntar sobre este texto.")
                    # --- FIN DEL "HACK" DE RAG ---

                    # 2. RESUMEN CON GPT-4
                    st.subheader("2. Resumen y Puntos Clave (GPT-4)")
                    system_prompt = """
                                    Eres "EstudIA", un asistente académico experto, amigable y perspicaz.
                                    Tu misión es leer la transcripción de una clase (de cualquier materia, desde Cálculo hasta Filosofía) y ayudar al estudiante a entender lo esencial.

                                    Por favor, responde SIEMPRE usando la siguiente estructura Markdown. Sé conciso y ve al grano:

                                    **Resumen:**
                                    (Aquí va tu resumen conciso, idealmente en 3 frases.)

                                    **Puntos Clave (los 5 conceptos más importantes):**
                                     1.  [Punto clave 1]
                                     2.  [Punto clave 2]
                                     3.  [Punto clave 3]
                                     4.  [Punto clave 4]
                                     5.  [Punto clave 5]

                                        (No añadas saludos ni despedidas, solo la estructura.)
                                            """
                    summary_response = client.chat.completions.create(
                        model="gpt-4-turbo",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": transcript_text}
                        ]
                    )
                    summary_text = summary_response.choices[0].message.content
                    st.success("¡Resumen generado!")
                    st.markdown(summary_text)

                except Exception as e:
                    st.error(f"Ocurrió un error: {e}")

# --- PESTAÑA 2: TUTOR ESTUDIA ---
with tab2:
    st.header("🧑‍🏫 Funciones 2 y 3: Chat Interactivo")
    st.write("Demuestra cómo el tutor guía, se adapta, o chatea sobre un texto específico.")

    # --- SELECTOR DE MODO (¡AHORA CON 3 MODOS!) ---
    demo_mode = st.selectbox(
        "Selecciona el Modo de Demo:",
        (
            "Modo: Alumno Nuevo (Tutor guía)", 
            "Modo: Chat con tu Pitch (RAG)" # <- ¡LA NUEVA OPCIÓN!
        )
    )
    
    # --- LÓGICA DE PROMPTS PARA MODO 1 y 2 ---
    PROMPT_ALUMNO_NUEVO = """
    Eres "EstudIA", un tutor socrático experto, tienes prohibido dar las respuestas, solo guiar en el resultado de aprendizaje paso por paso a me dida que transcurre al conversación...
    IMPORTANTE: Cuando escribas fórmulas matemáticas, usa SIEMPRE la sintaxis LaTeX...
    """ # (Abreviado, es tu V4.4)
    
    PROMPT_PERFIL_MARIA = """
    Eres "EstudIA", un tutor socrático experto...
    ---
    **INFORMACIÓN CONFIDENCIAL DEL ESTUDIANTE (MODO DEMO):**
    * **Nombre:** María.
    * **Punto Débil Detectado:** ...Derivadas...
    ---
    IMPORTANTE: Cuando escribas fórmulas matemáticas, usa SIEMPRE la sintaxis LaTeX...
    """ # (Abreviado, es tu V4.4)

    # --- LÓGICA DEL "ROUTER" ---
    
    # MODO 1 y 2: El Chat Manual (Tu V4.4 "Golden" que ya funciona)
    if demo_mode in ("Modo: Alumno Nuevo (Tutor guía)", "Modo: Perfil 'María' (DEMO)"):
        
        if demo_mode == "Modo: Perfil 'María' (DEMO)":
            system_prompt_a_usar = PROMPT_PERFIL_MARIA
            st.warning("Perfil 'María' cargado (débil en derivadas). El bot ahora se adaptará a ella.")
            session_key = "chat_history_maria" 
        else:
            system_prompt_a_usar = PROMPT_ALUMNO_NUEVO
            st.success("Perfil 'Alumno Nuevo' cargado. El bot actuará como un tutor socrático general.")
            session_key = "chat_history_nuevo"
        
        if session_key not in st.session_state:
            st.session_state[session_key] = []
        chat_history = st.session_state[session_key]

        try:
            # (Aquí va TODO tu código de chat manual de la V4.4)
            # --- MOSTRAR HISTORIAL (Modo 1 y 2) ---
            for msg in chat_history:
                msg_type = "user" if msg["role"] == "user" else "assistant"
                with st.chat_message(msg_type):
                    st.markdown(msg["content"]) # Usamos markdown

            # --- INPUT DEL USUARIO (Modo 1 y 2) ---
            if user_input := st.chat_input("Escribe tu duda socrática aquí..."):
                st.chat_message("user").write(user_input)
                chat_history.append({"role": "user", "content": user_input})
                
                with st.chat_message("assistant"):
                    with st.spinner("EstudIA (Socrático) está pensando..."):
                        messages = [{"role": "system", "content": system_prompt_a_usar}]
                        messages.extend(chat_history) # Más limpio
                        
                        response = client.chat.completions.create(
                            model="gpt-4-turbo",
                            messages=messages
                        )
                        response_text = response.choices[0].message.content
                        
                        # Tu hack de LaTeX
                        response_text = response_text.replace('\\[', '$$').replace('\\]', '$$')
                        response_text = response_text.replace('\\(', '$').replace('\\)', '$')
                        
                        st.markdown(response_text) 
                        chat_history.append({"role": "assistant", "content": response_text})
                        st.session_state[session_key] = chat_history
        
        except Exception as e:
            st.error(f"Error al inicializar el chat socrático: {e}")

    # MODO 3: El Chat RAG (¡La Nueva Función 5!)
    elif demo_mode == "Modo: Chat con tu Pitch (RAG)":
        
        st.info("Modo RAG: El bot ahora solo responderá basándose en el texto de la Pestaña 1.")
        
        # 1. Chequeamos si la transcripción existe
        if "pitch_transcrito" not in st.session_state:
            st.error("¡Error! Primero debes 'Transcribir y Resumir' un audio en la Pestaña 1.")
            st.stop()

        # 2. Construimos (o cargamos desde caché) la cadena RAG
        transcript_text = st.session_state["pitch_transcrito"]
        try:
            # ¡Llamamos a nuestro motor RAG!
            rag_chain = build_rag_chain(transcript_text, API_KEY)
            
            if rag_chain is None:
                st.error("Error al construir el motor RAG. Revisa los logs.")
                st.stop()
            
            # 3. Inicializamos la memoria de chat RAG
            session_key_rag = "chat_history_rag"
            if session_key_rag not in st.session_state:
                st.session_state[session_key_rag] = []
            
            chat_history_rag = st.session_state[session_key_rag]

            # 4. Mostrar historial RAG
            for msg in chat_history_rag:
                msg_type = "user" if msg["role"] == "user" else "assistant"
                with st.chat_message(msg_type):
                    st.markdown(msg["content"])

            # 5. Input del usuario RAG
            if user_input := st.chat_input("Pregunta sobre el pitch aquí..."):
                st.chat_message("user").write(user_input)
                
                with st.chat_message("assistant"):
                    with st.spinner("EstudIA (RAG) está buscando en el pitch..."):
                        
                        # ¡Aquí llamamos a la cadena RAG!
                        # Para el MVP, la memoria no es crucial, así que la enviamos simple
                        response = rag_chain.invoke({
                            "input": user_input
                        })
                        
                        response_text = response
                        st.markdown(response_text)
                        
                        # Actualizamos la memoria RAG (simple)
                        st.session_state[session_key_rag].append({"role": "user", "content": user_input})
                        st.session_state[session_key_rag].append({"role": "assistant", "content": response_text})

        except Exception as e:
            st.error(f"Error al construir la cadena RAG: {e}")
            st.error("El fantasma de LangChain atacó de nuevo. Revisa la terminal.")