"""
TODO: chat history implemented according to https://python.langchain.com/v0.2/docs/tutorials/qa_chat_history/
    However, I think max_new_token for the history_chain llm should be decreased.
    also, make some faulty formulated prompt when there is no history yet.
TODO: llama doesn't answer properly. maybe modify the prompt?
"""

try:
    __import__("pysqlite3")
    import pysqlite3
    import sys

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except:
    pass

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface.llms import HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage

import os
import time
import random
import shutil
from dotenv import load_dotenv

AI_MSG_TAG = "ai"
USER_MSG_TAG = "human"

LLM_MODELS_REPO = {
    "Mixtral-8x7B-Instruct-v0.1": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "Mistral-7B-Instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    # "Meta-Llama-3-8B-Instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "google/gemma-1.1-7b-it": "google/gemma-1.1-7b-it",
}

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


def get_user_doc():
    if st.session_state.doc_type == "PDF":
        pdf = st.file_uploader("Upload a PDF file to start:", type="pdf")
        if not pdf:
            return
    elif st.session_state.doc_type == "Web URL":
        url = st.text_input("Insert URL and apply:", placeholder="Insert URL")
        if not url:
            return
    elif st.session_state.doc_type == "Text":
        txt = st.text_area("Insert text and apply:", placeholder="Insert Text")
        if not txt:
            return
    else:
        return

    def lets_go_clicked():
        tmp_folder = os.path.join("./tmp", generate_random_folder_name())
        os.makedirs(tmp_folder)

        if st.session_state.doc_type == "PDF":
            doc_path = os.path.join(tmp_folder, "doc.pdf")
            with open(doc_path, "wb") as file:
                file.write(pdf.getvalue())
        elif st.session_state.doc_type == "Web URL":
            doc_path = url
        elif st.session_state.doc_type == "Text":
            doc_path = os.path.join(tmp_folder, "doc.txt")
            with open(doc_path, "w", encoding="utf-8") as file:
                file.write(txt)

        st.session_state.tmp_folder = tmp_folder
        st.session_state.doc_path = doc_path

    st.button("Let's Go!", on_click=lets_go_clicked)


def get_retriever():
    if st.session_state.doc_type == "PDF":
        loader = PyPDFLoader(st.session_state.doc_path)
    elif st.session_state.doc_type == "Web URL":
        loader = WebBaseLoader(web_path=st.session_state.doc_path)
    elif st.session_state.doc_type == "Text":
        loader = TextLoader(st.session_state.doc_path, encoding="utf-8")

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

    return vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})


def insert_message(role: str, msg: str, display=True):
    st.session_state.messages.append(msg_dict(role, msg))
    if display:
        with st.chat_message(role):
            st.markdown(stylized_msg(role, msg), unsafe_allow_html=True)


def insert_message_stream(stream):
    with st.chat_message(AI_MSG_TAG):
        message_placeholder = st.empty()
        full_response = ""
        st.session_state.messages.append(msg_dict(AI_MSG_TAG, full_response))

        for response in stream:
            full_response += response
            st.session_state.messages[-1]["content"] = full_response
            message_placeholder.markdown(stylized_msg(AI_MSG_TAG, full_response + "▌"), unsafe_allow_html=True)

        st.session_state.messages[-1]["content"] = full_response
        message_placeholder.markdown(stylized_msg(AI_MSG_TAG, full_response), unsafe_allow_html=True)


def stylized_msg(role: str, msg: str):
    return f"<div class='message-{role}'>{msg}</div>"


def display_all_messages():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(stylized_msg(message["role"], message["content"]), unsafe_allow_html=True)


def get_user_prompt():
    if user_prompt := st.chat_input("Your prompt"):
        insert_message(USER_MSG_TAG, user_prompt)
        return user_prompt
    return None


def merge_splits(splits):
    return "\n\n".join(f"{i}.{s.page_content}" for i, s in enumerate(splits))


def get_rag_chain():
    rag_chain = (
        {"context": st.session_state.retriever | merge_splits, "question": RunnablePassthrough()}
        | st.session_state.prompt_template
        | get_llm()
        | StrOutputParser()
    )
    return rag_chain


def get_history_aware_chain():
    contextualize_q_system_prompt = (
        "You are given a chat history and the last user question "
        "which might reference context in the chat history. "
        "You have to formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate the last question if needed and otherwise return the question as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "the last question: {question}"),
            ("ai", "Reformulated last question:"),
        ]
    )

    history_chain = contextualize_q_prompt | get_llm() | StrOutputParser()

    return history_chain


def get_llm():
    model_name = st.session_state.llm_model
    if model_name in st.session_state.llm_models_cache:
        return st.session_state.llm_models_cache[model_name]
    llm = HuggingFaceEndpoint(
        repo_id=LLM_MODELS_REPO[st.session_state.llm_model],
        huggingfacehub_api_token=os.environ["HF_API_KEY"],
        task="text-generation",
        streaming=True,
        # temperature=0.9,
        max_new_tokens=700,
        # top_p=0.95,
        # repetition_penalty=1.0,
    )
    st.session_state.llm_models_cache[model_name] = llm
    return llm


def get_prompt_template():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are an assistant for question-answering tasks.
                Use the following pieces of retrieved context to answer the question.
                If you don't know the answer, just say that you don't know.
                Use three sentences maximum and keep the answer concise.
                Context: {context}
                """,
            ),
            ("human", "Question: {question}"),
            ("ai", "Answer:"),
        ]
    )

    return prompt


def get_chat_history():
    chat_history = []
    # if there is less than 3 messages, it means that the first message is ai's and the second one is user's fresh message.
    # so there is no chat history between ai and user yet.
    # so we return an empty list as the chat history
    if len(st.session_state.messages) < 3:
        return chat_history
    for msg in st.session_state.messages[:-1]:
        if msg["role"] == AI_MSG_TAG:
            chat_history.append(AIMessage(content=msg["content"]))
        elif msg["role"] == USER_MSG_TAG:
            chat_history.append(HumanMessage(content=msg["content"]))
        else:
            raise ValueError(f"role {msg['role']} is not valid.")
    return chat_history


def main():
    cleanup_tmp_folder()

    # Set the page configuration
    st.set_page_config(
        page_title="DocDialogue",
        page_icon="🤖",
    )

    # set page style
    with open("./src/styles.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # set page title
    st.markdown(
        """
        <span style='font-size: 48px;font-weight: bold;'>🤖 DocDialogue</span>
        <span style='font-size: 20px;font-weight: normal;margin-left: 10px;'>Chat with your document!</span>
        """,
        unsafe_allow_html=True,
    )

    # add github link
    st.sidebar.markdown(
        """
        <a href="https://github.com/hmralavi/DocDialogue" class="github-link" target="_blank">
            <svg height="24" width="24" aria-hidden="true" viewBox="0 0 16 16" version="1.1" data-view-component="true" fill="white">
                <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.58.82-2.13-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.55.82 1.26.82 2.13 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.19 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
            </svg>
            GitHub
        </a>
        """,
        unsafe_allow_html=True,
    )

    if "tmp_folder" not in st.session_state:
        # doc type option
        st.selectbox(
            "Document type:",
            ["Select your document type", "PDF", "Web URL", "Text"],
            index=0,
            key="doc_type",
        )
        if st.session_state.doc_type != "Select your document type":
            get_user_doc()
        return

    if "llm_model" not in st.session_state:
        st.session_state.llm_model = list(LLM_MODELS_REPO)[0]

    if "llm_models_cache" not in st.session_state:
        st.session_state.llm_models_cache = {}

    if "retriever" not in st.session_state:
        st.session_state.retriever = get_retriever()

    if "prompt_template" not in st.session_state:
        st.session_state.prompt_template = get_prompt_template()

    if "rag_chain" not in st.session_state:
        st.session_state.history_aware_chain = get_history_aware_chain()
        st.session_state.rag_chain = get_rag_chain()

    # page side bar (model selection options)
    def llm_model_changed_callback():
        st.session_state.history_aware_chain = get_history_aware_chain()
        st.session_state.rag_chain = get_rag_chain()

    st.sidebar.markdown(
        "<span style='font-size: 36px;font-weight: bold;'>Options:</span>",
        unsafe_allow_html=True,
    )

    st.sidebar.radio(
        "Select LLM Model:",
        list(LLM_MODELS_REPO),
        index=0,
        on_change=llm_model_changed_callback,
        key="llm_model",
    )

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
        if len(st.session_state.messages) >= 3:
            prompt = st.session_state.history_aware_chain.invoke(
                {"question": prompt, "chat_history": get_chat_history()}
            )
            print("-----reformulated prompt-------")
            print(prompt)
        response_stream = st.session_state.rag_chain.stream(prompt)
        insert_message_stream(response_stream)


if __name__ == "__main__":
    main()
