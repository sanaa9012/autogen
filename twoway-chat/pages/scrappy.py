import os 
import requests
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.vectorstores import FAISS 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from jinja2 import Template

# Load API keys
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
JINA_API = os.getenv("JINA_API")

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

@st.cache_data
def scrap_site(url):
    response = requests.get(f"{JINA_API}/{url}")
    if response.status_code == 200:
        return response.text
    else:
        return f"Error: {response.status_code}, {response.text}"

@st.cache_resource
def get_vector_store_scrappy(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""])
    text_chunks = text_splitter.split_text(text)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore_scrappy = FAISS.from_texts(text_chunks, embeddings)
    vectorstore_scrappy.save_local("vectorstore_scrappy")
    return vectorstore_scrappy

def chat_bot_scrappy(user_query):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("vectorstore_scrappy", embeddings, allow_dangerous_deserialization=True)
    retrieved_docs = vector_store.similarity_search(user_query, k=10)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(f"Context: {context}\nUser: {user_query}\nSummarize the answer using your own words and provide general insights.")
    return response.text

def render_message(template_str, message):
    template = Template(template_str)
    return template.render(MSG=message)

# ------ UI Setup ------
st.set_page_config(page_title="Scrappy", page_icon="👾")
st.write(css, unsafe_allow_html=True)
st.title("🤖Scrappy Chatbot")
st.markdown("Ask a question about the website you added!")

# Initialize session state for chat history
if "scrappy_chat_history" not in st.session_state:
    st.session_state.scrappy_chat_history = []

# User enters URL to scrape
url = st.text_input("Enter the URL of the website you want to scrape:")
if st.button("Scrape"):
    with st.spinner("Scraping..."):
        text = scrap_site(url)
        vectorstore = get_vector_store_scrappy(text)
        st.success("Scraped successfully!")

# User asks a question
user_query = st.text_input("Ask your question:")
if st.button("Ask"):
    if user_query:
        with st.spinner("Thinking..."):
            response = chat_bot_scrappy(user_query)
            st.session_state.scrappy_chat_history.append((user_query, response))  # Store chat in history
            # st.success("Response:")
    else:
        st.warning("Please enter a question!")

# ------ Display Chat History ------
for user_msg, bot_msg in reversed(st.session_state.scrappy_chat_history):
    # st.markdown(f"👤 **You:** {user_msg}")
    # st.markdown(f"🤖 **Bot:** {bot_msg}")
    # st.markdown("---")  # Separator
    st.markdown(user_template.replace("{{MSG}}", user_msg), unsafe_allow_html=True)
    st.markdown(bot_template.replace("{{MSG}}", bot_msg), unsafe_allow_html=True)