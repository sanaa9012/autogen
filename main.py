import asyncio
from autogen_core.models import UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
import streamlit as st
from jinja2 import Template
from dotenv import load_dotenv
import os
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Supabase credentials
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

if not url or not key:
    raise ValueError("Supabase URL and Key must be set in environment variables.")

# Initialize Supabase client
supabase: Client = create_client(url, key)

# Initialize AI Model Client
model_client = OpenAIChatCompletionClient(
    model="gemini-1.5-flash-8b",
    api_key=os.getenv("GOOGLE_API_KEY")
)

# Inject CSS for better UI
css = '''
<style>
.chat-message {
    padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex;
}
.chat-message.user {
    background-color: #475063;
}
.chat-message.bot {
    background-color: black;
}
.chat-message .avatar {
    width: 10%;
}
.chat-message .avatar img {
    max-width: 60px;
    max-height: 60px;
    border-radius: 50%;
    object-fit: cover;
}
.chat-message .message {
    width: auto;
    padding: 0 1rem;
    color: #fff;
}
</style>
'''
st.write(css, unsafe_allow_html=True)

# Define HTML templates for messages
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

# Function to render a message using a template
def render_message(template_str, message):
    template = Template(template_str)
    return template.render(MSG=message)

# Function to get response from AI model
async def get_response(user_input):
    response = await model_client.create([UserMessage(content=user_input, source="user")])
    return response.content.strip()

# Function to store chat messages in Supabase
def store_chat(user_msg, bot_msg):
    data = {
        "user_message": user_msg,
        "bot_response": bot_msg,
    }
    supabase.table("chat_history").insert(data).execute()

# Function to fetch chat history from Supabase
def fetch_chat_history():
    response = supabase.table("chat_history").select("*").order("id", desc=True).execute()
    return response.data if response.data else []

# Streamlit UI
st.header("💬 Chat with AI")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = fetch_chat_history()

# Input box for user query
user_query = st.text_input("Ask your question:")

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

if st.button("Ask"):
    if user_query:
        with st.spinner("Thinking..."):
            response_future = asyncio.run_coroutine_threadsafe(get_response(user_query), loop)
            response = response_future.result()

            # Store chat in Supabase
            store_chat(user_query, response)

            # Store chat in session state
            st.session_state.chat_history.insert(0, {"user_message": user_query, "bot_response": response})

# Display Chat History
for chat in st.session_state.chat_history:
    st.markdown(render_message(user_template, chat["user_message"]), unsafe_allow_html=True)
    st.markdown(render_message(bot_template, chat["bot_response"]), unsafe_allow_html=True)
