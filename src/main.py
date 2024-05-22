import os
from dotenv import load_dotenv
import time
import random
import shutil

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS


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

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=32)
    texts = text_splitter.split_documents(documents)

    embedding_model = HuggingFaceInferenceAPIEmbeddings(
        api_key=os.environ["HF_API_KEY"], model_name="sentence-transformers/all-MiniLM-l6-v2"
    )

    vectordb = FAISS.from_documents(
        documents=texts,
        embedding=embedding_model,
    )

    st.session_state.vectordb = vectordb


def retrieve_ai_response():
    # generate responses
    with st.chat_message("AI"):
        message_placeholder = st.empty()
        full_response = ""

        for response in mymodel(
            model=st.session_state.model,
            messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
            stream=True,
        ):
            # full_response += response.choices[0].delta.get("content", "")
            full_response += response
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "AI", "content": full_response})


def insert_ai_message(msg: str):
    def msg_generator():
        for c in msg:
            yield c
            time.sleep(0.01)

    # generate responses
    with st.chat_message("AI"):
        message_placeholder = st.empty()
        full_response = ""

        for response in msg_generator():
            full_response += response
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "AI", "content": full_response})


def display_messages():
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def get_user_prompt():
    if user_prompt := st.chat_input("Your prompt"):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)
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
        st.session_state.messages.append(
            {
                "role": "AI",
                "content": "Hello, this is your AI assistant. I will answer questions about your document.",
            }
        )

    if "tmp_folder" not in st.session_state:
        run_upload()
        return

    if "vectordb" not in st.session_state:
        run_embedding()

    display_messages()

    prompt = get_user_prompt()

    if prompt:
        similar_splits = st.session_state.vectordb.similarity_search(prompt, k=5)
        response = "Top 5 relative splits:\n\n"
        response += "\n\n".join([f"{i+1}. {sp.page_content}" for i, sp in enumerate(similar_splits)])
        insert_ai_message(response)


if __name__ == "__main__":
    main()
