#https://adasci.org/rag-with-milvus-vector-database-and-langchain/

#pip install langchain_community
#pip install langchain_core
import random
import time
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

#pip install langchain-huggingface
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
from langchain_community.document_loaders import PyPDFLoader

#%pip install -qU  langchain_milvus
from langchain_milvus import Milvus

# with open('./data/uber_2021.pdf','r') as txt:
#      context = txt.read()

loader = PyPDFLoader('./data/dados.pdf')
pages = loader.load_and_split()

vector_store = Milvus.from_documents(documents=pages,embedding=embeddings,connection_args={"uri":"./milvus_articles.db",},drop_old=True)

llm = Ollama(
    model='llama3',
    temperature=0
)

prompt = PromptTemplate(
    input_variables=["context","question"],
    template = """
    # INSTRUÇÃO
    Você é um atendente e deve esclarecer as dúvidas enviadas pelo usuário.
    Se a pergunta não estiver relacionada a Amamentação materna ou não estiver nos dados do contexto, responda amigavelmente que só pode ajudar com dúvidas nesse contexto.

    # CONTEXTO PARA RESPOSTAS
    {context}

    # PERGUNTA
    Pergunta: {question}
    """
)

retriever = vector_store.as_retriever()

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()} 
    | prompt
    | llm
    | StrOutputParser()

)

st.title("Chat Amamentação")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if pergunta:= st.chat_input("Faça uma pergunta sobre amamentação para o assistente"):
    with st.chat_message("user"):
        st.markdown(pergunta)
        st.session_state.messages.append({"role":"user","content":pergunta})

    resposta = rag_chain.invoke(pergunta)

    with st.chat_message("assistant"):
        st.markdown(resposta)
        #print(resposta)
        st.session_state.messages.append({"role":"assistant","content":str(resposta)})
    



    
    


# while True:

#     pergunta = input("Faça sua pergunta ao assistente: ")
#     resumo = rag_chain.invoke(pergunta)
#     print(f"Resposta: {resumo}")
#     print("------------------------------ \n\n")

