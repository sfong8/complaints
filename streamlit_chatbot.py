import streamlit as st
from rag_functions import getContext, getResponse
def main():
    st.title("Complaint Query Chatbot")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("How can we assist you with your complaint?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        context = getContext(prompt.lower())
        response = getResponse(prompt.lower(),context)
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        with st.chat_message("assistant"):
            st.markdown(response)

if __name__ == "__main__":
    main()