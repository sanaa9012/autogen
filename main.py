import asyncio
from autogen_core.models import UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
import streamlit as st
from jinja2 import Template

# Initialize OpenAI Client
model_client = OpenAIChatCompletionClient(
    model="gemini-1.5-flash-8b",
    api_key="AIzaSyBQz63m0H2YXs3oNYMXTO-G5hmN8uRKTmk",
)

# Inject CSS for better UI
# background-color: #2b313e;
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

# Streamlit UI
st.write("Use the sidebar to navigate through different pages.")

# Chatbot Section
st.header("💬 Chat with AI")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input box for user query
user_query = st.text_input("Ask your question:")

if st.button("Ask"):
    if user_query:
        with st.spinner("Thinking..."):
            response = asyncio.run(get_response(user_query))
            st.session_state.chat_history.append((user_query, response))  # Store chat in history
            
        # # Render chat messages using the template
        # st.markdown(render_message(user_template, user_query), unsafe_allow_html=True)
        # st.markdown(render_message(bot_template, response), unsafe_allow_html=True)
    else:
        st.warning("Please enter a question!")

# # ------ Display Chat History ------
# st.subheader("Chat History")
for user_msg, bot_msg in reversed(st.session_state.chat_history):
    st.markdown(render_message(user_template, user_msg), unsafe_allow_html=True)
    st.markdown(render_message(bot_template, bot_msg), unsafe_allow_html=True)