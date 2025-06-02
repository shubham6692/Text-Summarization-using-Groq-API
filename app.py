import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document

## Streamlit
st.set_page_config(page_title="Summarize content from Youtube video or Website")
st.title("Summarize content from Youtube video or Website")
st.subheader("Summarize URL")

with st.sidebar:
    groq_api_key=st.text_input("Grok API Key", value="", type="password")
    
input_url=st.text_input("URL", label_visibility="collapsed")

prompt_template= """
Provide a summary of the following content in 200 words.
Add a motivational title, start the precise summary with an introduction and provide summary with bullet points
Content:{text}
"""

prompt=PromptTemplate(input_variables=["text"], template=prompt_template)

if st.button("Summarize the content from Youtube or Website"):
    ## validate inputs
    if not groq_api_key.strip() or not input_url.strip():
        st.error("Please provide the required information")
    elif not validators.url(input_url):
        st.error("Please enter a valid URL")
    else:
        try:
            with st.spinner("Waiting...."):
                # load the website/YT video data
                if "youtube.com" in input_url:
                    #loader=YoutubeLoader.from_youtube_url(input_url,add_video_info=True)
                    video_id = input_url.split("v=")[-1]
                    transcript = YouTubeTranscriptApi.get_transcript(video_id=video_id)
                    text = " ".join([entry['text'] for entry in transcript])
                    docs = [Document(page_content=text)]                
                else:
                    loader=UnstructuredURLLoader(urls=[input_url], ssl_verify=False,
                    headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                    docs=loader.load()
                
                llm= ChatGroq(groq_api_key=groq_api_key, model="gemma2-9b-it")
                
                ### Chain for Summariztion
                chain=load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary=chain.run(docs)
                
                st.success(output_summary)
        
        except Exception as e:
            st.exception(f"Exception:{e}")
                