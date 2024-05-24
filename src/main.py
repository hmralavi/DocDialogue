"""
TODO: option for using different llm models
TODO: implementing chat history https://python.langchain.com/v0.2/docs/tutorials/qa_chat_history/
"""

try:
    __import__("pysqlite3")
    import pysqlite3
    import sys

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except:
    pass

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_huggingface.llms import HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain import hub

import os
import time
import random
import shutil
from dotenv import load_dotenv

AI_MSG_TAG = "assistant"
USER_MSG_TAG = "user"

load_dotenv()


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


def get_retriever():
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
        persist_directory=os.path.join(st.session_state.tmp_folder, "chroma"),
    )

    return vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})


def insert_message(role: str, msg: str, display=True):
    st.session_state.messages.append(msg_dict(role, msg))
    if display:
        with st.chat_message(role):
            st.markdown(msg)


def insert_message_stream(stream):
    with st.chat_message(AI_MSG_TAG):
        message_placeholder = st.empty()
        full_response = ""
        st.session_state.messages.append(msg_dict(AI_MSG_TAG, full_response))

        for response in stream:
            full_response += response
            st.session_state.messages[-1]["content"] = full_response
            message_placeholder.markdown(full_response + "▌")

        st.session_state.messages[-1]["content"] = full_response
        message_placeholder.markdown(full_response)


def display_all_messages():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def get_user_prompt():
    if user_prompt := st.chat_input("Your prompt"):
        insert_message(USER_MSG_TAG, user_prompt)
        return user_prompt
    return None


def merge_splits(splits):
    return "\n\n".join(f"{i}.{s.page_content}" for i, s in enumerate(splits))


def get_chain():
    rag_chain = (
        {"context": get_retriever() | merge_splits, "question": RunnablePassthrough()}
        | get_prompt_template()
        | get_llm()
        | StrOutputParser()
    )
    return rag_chain


def get_llm():
    llm = HuggingFaceEndpoint(
        # repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        huggingfacehub_api_token=os.environ["HF_API_KEY"],
        task="text-generation",
        streaming=True,
        # temperature=0.9,
        # max_new_tokens=512,
        # top_p=0.95,
        # repetition_penalty=1.0,
    )
    return llm


def get_prompt_template():
    return hub.pull("rlm/rag-prompt")

    tmp = """
    [INST] You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. [/INST]

    [USER] Question: {question} [/USER]

    [META] Context: {context} [/META]

    [ASSISTANT] Answer:
    """
    return PromptTemplate(template=tmp, input_variables=["question", "context"])


def main():
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

    if "tmp_folder" not in st.session_state:
        run_upload()
        return

    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = get_chain()

    if "messages" not in st.session_state:
        st.session_state.messages = []
        insert_message(
            AI_MSG_TAG,
            "Hello, this is your AI assistant. I will answer questions about your document.",
            False,
        )

    display_all_messages()

    prompt = get_user_prompt()

    if prompt:
        response_stream = st.session_state.rag_chain.stream(prompt)
        insert_message_stream(response_stream)


if __name__ == "__main__":
    main()
