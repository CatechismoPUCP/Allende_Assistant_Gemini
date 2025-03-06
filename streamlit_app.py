import os
import streamlit as st
import google.generativeai as genai
import time
from datetime import datetime

# --- Constants and Configurations ---
API_KEY = "AIzaSyAhJyraVxu3WTX_WIaMKAu744DdgYlad00"  # Consider using st.secrets
PROMPTS_FOLDER = "Character_Prompts"
ICONS_FOLDER = "character_ico"
DEFAULT_SAFETY = "Low"  # Safety setting
MODEL_NAME = "gemini-1.5-pro"

# --- Configure Gemini ---
os.environ["API_KEY"] = API_KEY  # Or use st.secrets for better security
genai.configure(api_key=os.environ["API_KEY"])

# --- Utility Functions ---
@st.cache_data
def get_character_files():
    return [f.split('.')[0] for f in os.listdir(PROMPTS_FOLDER) if f.endswith('.txt')]

@st.cache_data
def load_character_prompt(selected_character):
    character_prompt_file = os.path.join(PROMPTS_FOLDER, f"{selected_character}.txt")
    with open(character_prompt_file, 'r', encoding='utf-8') as f:
        return f.read()

def process_response(response_text):
    parts = {}
    for key in ["question_analysis", "answer", "suggested_documents", "citations"]:
        start_tag, end_tag = f"<{key}>", f"</{key}>"
        start_idx, end_idx = response_text.find(start_tag), response_text.find(end_tag)
        parts[key] = response_text[start_idx + len(start_tag):end_idx].strip() if start_idx != -1 and end_idx != -1 else ""
    return parts

def initialize_session_state():
    defaults = {
        'chat_history': [],
        'user_input': "",
        'chat_session': None,
        'selected_character': None,
        'response_time': 0
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# --- Chat Session Creation ---
def create_new_chat_session(selected_character):
    system_prompt = load_character_prompt(selected_character).replace("{{USER_MESSAGE}}", "{user_input}")
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 1,
        "max_output_tokens": 10192,
    }
    safety_settings = {
        "harassment": "BLOCK_NONE", #Example
        "hate_speech": "BLOCK_NONE", #Example
        "sexually_explicit": "BLOCK_NONE", #Example
        "dangerous_content": "BLOCK_NONE", #Example
    }

    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config=generation_config,
        safety_settings="BLOCK_NONE",
        system_instruction=system_prompt
    )
    return model.start_chat(history=[])


# --- Main Application ---
def main():
    st.title("Allende Archives AI Assistant")
    initialize_session_state()

    character_files = get_character_files()
    selected_character = st.selectbox("Choose a historical character:", character_files, key="char_select")

    if st.session_state.selected_character != selected_character:
        st.session_state.chat_session = None
        st.session_state.chat_history = []
        st.session_state.selected_character = selected_character
        st.session_state.response_time = 0


    # --- Chat Interface ---
    with st.container():
        for role, message in st.session_state.chat_history:
            if role == "You":
                st.markdown(f"<span style='color: blue;'>You:</span> {message}", unsafe_allow_html=True)
            else:
                st.markdown(f"<span style='color: green;'>Archivist:</span> {message['answer']}", unsafe_allow_html=True)
                with st.expander("Show Analysis and Sources"):
                    st.markdown("### Question Analysis\n" + message["question_analysis"])
                    st.markdown("### Suggested Documents\n" + message["suggested_documents"])
                    st.markdown("### Citations\n" + message["citations"])

    # --- Input and Response Handling ---
    with st.container():
        user_input = st.text_input("Enter your message:", key="input_field")
        if st.button("Send") and user_input:
            if st.session_state.chat_session is None:
                st.session_state.chat_session = create_new_chat_session(selected_character)

            start_time = time.time()  # Start timing
            with st.spinner("Generating response..."):
                response = st.session_state.chat_session.send_message(user_input)
            st.session_state.response_time = time.time() - start_time  # End timing

            response_dict = process_response(response.text)
            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append((selected_character, response_dict))
            st.rerun()

    # --- Display Response Time ---
    if st.session_state.response_time > 0:
        st.write(f"Response time: {st.session_state.response_time:.2f} seconds")

    st.markdown("---")
    st.write("Powered by Gemini AI")

if __name__ == "__main__":
    main()