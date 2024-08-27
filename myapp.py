import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Check if the API key is loaded properly
if not api_key:
    st.error("Google API Key not found. Please check your .env file.")
else:
    genai.configure(api_key=api_key)

# Cache the model loading process to avoid reloading on each request
@st.cache_resource 
def get_llama_response(input_text, no_words, blog_style):
    """Generate a blog using Google Generative AI."""
    
    # Initialize the Google Generative AI model
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    # Prompt template for the blog generation
    template = """
        Write a blog for the {blog_style} job profile on the topic "{input_text}"
        within {no_words} words.
    """
    
    # Format the prompt
    formatted_prompt = template.format(blog_style=blog_style, input_text=input_text, no_words=no_words)
    
    # Generate the response from the AI model
    response = llm(formatted_prompt)

    # Extract the generated text from the response
    if response and 'generations' in response:
        generated_text = response['generations'][0]['text']
    else:
        generated_text = "<No response from the model>"
    
    print(generated_text)
    return generated_text

# Streamlit page setup
st.set_page_config(page_title="Generate Blogs",
                   page_icon='🤖',
                   layout='centered',
                   initial_sidebar_state='collapsed')

st.header("Generate Blogs 🤖")

# User input fields
input_text = st.text_input("Enter the Blog Topic")

col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input('No of Words')
with col2:
    blog_style = st.selectbox('Writing the blog for', ('Researchers', 'Data Scientist', 'Common People'), index=0)
    
submit = st.button("Generate")

# Generate and display the blog based on user input
if submit:
    if not input_text or not no_words:
        st.error("Please fill in all the fields.")
    else:
        st.write(get_llama_response(input_text, no_words, blog_style))
