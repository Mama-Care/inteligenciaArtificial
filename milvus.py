#pip install langchain-huggingface
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

#%pip install -qU  langchain_milvus
from langchain_milvus import Milvus

from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def initialize_milvus():


    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

    #loader = PyPDFLoader('./data/dados.pdf')

    # Configurando o TextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=300)

    documents = []

    # Listar todos os arquivos PDF na pasta './data'
    for file_name in os.listdir('./data'):
        if file_name.endswith('.pdf'):  # Verifica se o arquivo é um PDF
            file_path = os.path.join('./data', file_name)
            print(f"Carregando arquivo: {file_path}")
            
            # Carrega o conteúdo do PDF e divide em partes
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load_and_split(text_splitter=text_splitter))  # Adiciona as páginas à lista

    vector_store = Milvus.from_documents(documents=documents, embedding=embeddings, connection_args={"uri":"./milvus_articles.db",}, drop_old=True)

    return vector_store