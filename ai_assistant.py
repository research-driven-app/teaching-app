import streamlit as st

from llama_index.core import VectorStoreIndex, ServiceContext, Document
from llama_index.llms.openai import OpenAI
import openai
from llama_index.core import SimpleDirectoryReader


import os
import shutil

def copy_and_rename_files(src_folder_path, dest_folder_path, new_name_suffix):
    # Ensure the destination folder exists
    os.makedirs(dest_folder_path, exist_ok=True)
    
    # Iterate over all files in the source folder
    for index, filename in enumerate(os.listdir(src_folder_path)):
        src_file_path = os.path.join(src_folder_path, filename)
        
        # Skip directories
        if os.path.isdir(src_file_path):
            continue
        
        # Construct the new file name
        new_file_name = filename + new_name_suffix
        dest_file_path = os.path.join(dest_folder_path, new_file_name)
        
        # Copy the file to the new location with the new name
        shutil.copy2(src_file_path, dest_file_path)
        

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
            
            src_folder_path = 'data\encrypted_PDFs'
            dest_folder_path = 'data\paper_PDFs'
            new_name_suffix = '.pdf'

            copy_and_rename_files(src_folder_path, dest_folder_path, new_name_suffix)

            reader = SimpleDirectoryReader(input_dir="./data/paper_PDFs", recursive=True)
            docs = reader.load_data()

            service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.0, system_prompt=syst_prmpt))
            
            index = VectorStoreIndex.from_documents(docs, service_context=service_context)

            return index