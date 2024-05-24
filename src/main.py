import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_chroma import Chroma

import os
import time
import random
import shutil
from dotenv import load_dotenv

AI_MSG_TAG = "assistant"
USER_MSG_TAG = "user"


def msg_dict(role: str, content: str):
    return {"role": role, "content": content}


def generate_random_folder_name():
    timestamp = int(time.time())
    random_str = "".join([str(random.randint(1, 9)) for _ in range(4)])
    return f"{timestamp}_{random_str}"


def cleanup_tmp_folder():
    tmp_path = "./tmp"
    if not os.path.exists(tmp_path):
        return
    folders = os.listdir(tmp_path)
    folders.sort()
    folders = folders[::-1]
    for i, fname in enumerate(folders):
        if i < 10:
            continue
        shutil.rmtree(os.path.join(tmp_path, fname))


def run_upload():
    uploaded_file = st.file_uploader("Upload a PDF file to start:", type="pdf")
    if not uploaded_file:
        return

    def upload_clicked():
        tmp_folder = os.path.join("./tmp", generate_random_folder_name())
        os.makedirs(tmp_folder)

        doc_path = os.path.join(tmp_folder, "doc.pdf")
        with open(doc_path, "wb") as file:
            file.write(uploaded_file.getvalue())
            st.session_state.tmp_folder = tmp_folder
            st.session_state.doc_path = doc_path

    st.button("Let's Go!", on_click=upload_clicked)


def run_embedding():
    loader = PyPDFLoader(st.session_state.doc_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    embedding_model = HuggingFaceInferenceAPIEmbeddings(
        api_key=os.environ["HF_API_KEY"],
        model_name="sentence-transformers/all-MiniLM-l6-v2",
    )

    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embedding_model,
    )

    st.session_state.vectordb = vectordb


def insert_message(role: str, msg: str, display=True):
    st.session_state.messages.append(msg_dict(role, msg))
    if display:
        with st.chat_message(role):
            st.markdown(msg)


def display_all_messages():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def get_user_prompt():
    if user_prompt := st.chat_input("Your prompt"):
        insert_message(USER_MSG_TAG, user_prompt)
        return user_prompt
    return None


def main():
    load_dotenv()

    cleanup_tmp_folder()

    # Set the page configuration
    st.set_page_config(
        page_title="DocDialogue",
        page_icon="🤖",
    )

    # set page title
    st.markdown(
        """<span style='font-size: 48px;font-weight: bold;'>🤖 DocDialogue</span>
        <span style='font-size: 20px;font-weight: normal;margin-left: 10px;'>Chat with your document!</span>""",
        unsafe_allow_html=True,
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []
        insert_message(
            AI_MSG_TAG,
            "Hello, this is your AI assistant. I will answer questions about your document.",
            False,
        )

    if "tmp_folder" not in st.session_state:
        run_upload()
        return

    if "vectordb" not in st.session_state:
        run_embedding()

    display_all_messages()

    prompt = get_user_prompt()

    if prompt:
        similar_splits = st.session_state.vectordb.similarity_search(prompt, k=5)
        response = "Top 5 relative splits:\n\n"
        response += "\n\n".join([f"{i+1}. {sp.page_content}" for i, sp in enumerate(similar_splits)])
        insert_message(AI_MSG_TAG, response)


if __name__ == "__main__":
    main()
