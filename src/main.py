import streamlit as st


def main():
    st.title("Chat with a Large Language Model")

    # Text input box for user input
    user_input = st.text_input("You:", "")

    # Check if user has entered any input
    if user_input:
        # Generate response from the model based on user input
        response = generate_response(user_input)
        st.text_area("LLM:", value=response, height=150, max_chars=None)


def generate_response(user_input):
    # Generate response from the model
    response = f"{user_input}:LLM response"
    return response


if __name__ == "__main__":
    main()
