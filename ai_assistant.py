import streamlit as st

from llama_index.core import VectorStoreIndex, ServiceContext, Document
from llama_index.llms.openai import OpenAI
import openai
from llama_index.core import SimpleDirectoryReader

openai.api_key = st.secrets.openai_key



syst_prmpt = '''
"You are an expert on the paper 'Real-Time Brand Reputation Tracking Using Social Media' from the Journal of Marketing 2021.
and your job is to answer technical questions only adopting the provided context. 
Assume that all questions are related to the paper. 
Keep your answers technical and based on facts – do not hallucinate concept not described in the paper."
'''

@st.cache_resource(show_spinner=False)
def load_data():
    with st.sidebar:
        with st.spinner(text="Booting up the AI Assistant..."):
            reader = SimpleDirectoryReader(input_dir="./data/paper_PDFs", recursive=True)
            docs = reader.load_data()
            service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.0, system_prompt=syst_prmpt))
            index = VectorStoreIndex.from_documents(docs, service_context=service_context)
            return index