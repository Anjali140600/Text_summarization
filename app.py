import validators
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
import os
from dotenv import load_dotenv

load_dotenv()

# Streamlit app setup
st.set_page_config(page_title="LangChain: Summarize Text from YouTube or Website", page_icon=":robot_face:")
st.title("LangChain: Summarize Text from YouTube or Website")
st.subheader("Summarize URL")

# Sidebar for API key input
with st.sidebar:
    groq_api_key = st.text_input("Enter your Groq API Key", value="", type="password")

# URL input field
generic_url = st.text_input("Enter URL (YouTube or Website)", label_visibility="collapsed")

# Only create the LLM after key is entered
llm = None
if groq_api_key:
    llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)

# Prompt template for summarization
prompt_template = """
Provide a 300-word summary of the following content:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# Summarize button logic
if st.button("Summarize the content from YouTube or Website"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide both the API key and the URL.")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL.")
    else:
        try:
            with st.spinner("Fetching and summarizing content..."):
                # Load data depending on source
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0"}
                    )

                docs = loader.load()

                # ✅ Check if any text was extracted
                if not docs or not docs[0].page_content.strip():
                    st.warning("It appears the content couldn't be extracted from the URL. "
                               "This might happen if the site uses JavaScript or if YouTube has no transcript.")
                else:
                    # Summarization chain
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    output_summary = chain.run(docs)
                    st.success(output_summary)

        except Exception as e:
            st.error(f"Error: {e}")
