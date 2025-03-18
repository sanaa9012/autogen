import asyncio
from autogen_core.models import UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
import streamlit as st

model_client = OpenAIChatCompletionClient(
    model="gemini-1.5-flash-8b",
    api_key="AIzaSyBQz63m0H2YXs3oNYMXTO-G5hmN8uRKTmk",
)

async def test_model():
    response = await model_client.create([UserMessage(content="What is the capital of Yemen?", source="user")])
    print(response.content.strip())

asyncio.run(test_model())

st.title("🏠 Welcome to My Streamlit App!")
st.write("Use the sidebar to navigate through different pages.")

st.sidebar("Select a page above to navigate.")