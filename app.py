import streamlit as st
from langchain.prompts import PromptTemplate
from transformers import pipeline
from dotenv import load_dotenv
from huggingface_hub import login


login(token='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')



## Function to get response from LLaMA 2 model using Hugging Face pipeline

@st.cache_resource ###This way, the model will only be loaded once, and subsequent requests will reuse the same instance.
def getLLamaresponse(input_text, no_words, blog_style):

    ## Hugging Face Pipeline for LLaMA 2
    llm = pipeline(
        model="meta-llama/Meta-Llama-3.1-8B",
        task="text-generation",
        use_auth_token=True  # Assuming you have a Hugging Face account and API key set up
    )


    ## Prompt Template
    template = """
        Write a blog for {blog_style} job profile for a topic {input_text}
        within {no_words} words.
    """
    
    prompt = PromptTemplate(input_variables=["blog_style", "input_text", "no_words"], template=template)
    
    ## Generate the response from the LLaMA 2 model
    response = llm(prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words))

    generated_text = response[0]['generated_text'] if response else "<No response from the model>"
    
    print(generated_text)
    return generated_text


load_dotenv()
st.set_page_config(page_title="Generate Blogs",
                   page_icon='🤖',
                   layout='centered',
                   initial_sidebar_state='collapsed')

st.header("Generate Blogs 🤖")

input_text = st.text_input("Enter the Blog Topic")

## Creating two more columns for additional fields
col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input('No of Words')
with col2:
    blog_style = st.selectbox('Writing the blog for', ('Researchers', 'Data Scientist', 'Common People'), index=0)
    
submit = st.button("Generate")

## Final response
if submit:
    st.write(getLLamaresponse(input_text, no_words, blog_style))
