import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Inject CSS for chat messages
css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
</style>
'''

# Define HTML templates for chat messages
bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ43puwjMM4AUDfBHvZ9rhxriJO3c2N1JfjNSdERIptWd1Ts69jgcawbWdw076b8X0RYAY&usqp=CAU">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    return text_splitter.split_text(text)

# Function to create a vector store
def get_vectorstore_pdftalk(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
    vectorstore_pdftalk = FAISS.from_texts(text_chunks, embedding=embeddings)
    vectorstore_pdftalk.save_local("vectorstore_pdftalk")
    return vectorstore_pdftalk

# Function to create a conversation chain
def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=1)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)

# Function to handle user input
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.pdftalk_chat_history.append((user_question, response["answer"]))

# Streamlit UI
def main():
    st.set_page_config(page_title="PDF Talk", page_icon="📚")
    st.markdown(css, unsafe_allow_html=True)

    # Session state initialization
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "pdftalk_chat_history" not in st.session_state:
        st.session_state.pdftalk_chat_history = []

    st.header("📚 PDF Talk - Chat with Your Documents")
    
    # Sidebar for file upload
    with st.sidebar:
        st.subheader("Upload PDFs")
        pdf_docs = st.file_uploader("Upload PDF files and click 'Process':", accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                if raw_text.strip():
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore_pdftalk(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.success("PDF Processing Complete! You can start chatting.")
                else:
                    st.error("No text extracted from PDF. Please check your files.")

    # User chat input
    user_question = st.text_input("Ask something about your PDF:")
    if user_question:
        if st.session_state.conversation is None:
            st.error("Please upload PDFs and click 'Process' first.")
        else:
            handle_userinput(user_question)

    # Display chat history with formatted HTML templates
    for user_msg, bot_msg in reversed(st.session_state.pdftalk_chat_history):
        st.markdown(user_template.replace("{{MSG}}", user_msg), unsafe_allow_html=True)
        st.markdown(bot_template.replace("{{MSG}}", bot_msg), unsafe_allow_html=True)

if __name__ == "__main__":
    main()