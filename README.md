# DocDialogue

A simple web application created with streamlit and python that enables you to chat with your document!

This app employs Retrieval Augmented Generation (RAG) technique combined with open-source LLM models.

# Introduction

Retrieval-Augmented Generation (RAG) is a technique that enhances the capabilities of a language model (LLM) by integrating a retrieval mechanism to access external data sources, thus allowing the model to generate responses based on the most up-to-date and relevant information. 

By combining RAG with LLMs, you create a powerful system capable of delivering highly accurate, relevant, and timely information, significantly expanding the utility and effectiveness of language models in dynamic information environments.

# How to run the app

1. Open Terminal/Command Prompt: Open your terminal (on macOS/Linux) or Command Prompt (on Windows).

2. Navigate to the Desired Directory: Use the cd command to navigate to the directory where you want to clone the repository. For example:

    `cd path/to/your/directory`

3. Run the git clone Command:

    `git clone https://github.com/hmralavi/DocDialogue`

4. Navigate to the Cloned Repository:

    `cd DocDialogue`

5. Install python dependencies:

    `pip install -r requirements.txt`

6. run the app:

    `streamlit run src/main.py`

## NOTE

You need a HuggingFace API token to run this app. Don't worry, the token is free!

After receiving the api token, save it in a `.env` file inside the cloned repository. use this name inside the `.env` file:

`HF_API_KEY = "your huggingface api key"`