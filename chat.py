#https://adasci.org/rag-with-milvus-vector-database-and-langchain/

#pip install langchain_community
#pip install langchain_core
import random
import time
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st

#pip install langchain-huggingface
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
from langchain_community.document_loaders import PyPDFLoader

#%pip install -qU  langchain_milvus
from langchain_milvus import Milvus

loader = PyPDFLoader('./data/dados.pdf')

# Configurando o TextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=300)

# Carregando e dividindo o PDF usando o TextSplitter
pages = loader.load_and_split(text_splitter=text_splitter)

vector_store = Milvus.from_documents(documents=pages, embedding=embeddings, connection_args={"uri":"./milvus_articles.db",}, drop_old=True)

llm = Ollama(
    model='llama3',
    temperature=0
)

def getBetterPromptByDistance(distance):
    
    prompt = ""

    if 0.8 <= distance <= 1:

        prompt = PromptTemplate(
            input_variables=["context","question"],
            template = """
            # INSTRUÇÃO
            Você é um atendente e deve esclarecer as dúvidas enviadas pelo usuário.
            Junto com sua resposta deixe claro que ela não é uma resposta que pode conter inconsistências

            # CONTEXTO PARA RESPOSTAS
            {context}

            # PERGUNTA
            Pergunta: {question}
            """
        )
    
    else:

        prompt = PromptTemplate(
            input_variables=["context","question"],
            template = """
            # INSTRUÇÃO
            Você é um atendente e deve esclarecer as dúvidas enviadas pelo usuário.

            # CONTEXTO PARA RESPOSTAS
            {context}

            # PERGUNTA
            Pergunta: {question}
            """
        )

    return prompt

#retriever = vector_store.as_retriever()

def calculation_distance(pergunta):
    results = vector_store.similarity_search_with_score(pergunta, k=1)
    distance = 0
    for content in results:
        distance = content[-1]
    return distance

def get_context(pergunta):
    # Usa o similarity_search para recuperar o contexto mais relevante
    results = vector_store.similarity_search(pergunta, k=1)
    # Retorna o conteúdo do primeiro resultado
    return results[0].page_content if results else ""

def process_question(question):

    distance = calculation_distance(question)
    resposta = ""

    if distance > 1:
        resposta = "Parece que essa pergunta está fora do meu tema principal, que é amamentação. Se precisar de informações ou apoio sobre amamentação, estou aqui para ajudar no que for possível!"

    else:

        # Recupera o contexto mais relevante
        context = get_context(question)

        prompt = getBetterPromptByDistance(distance)

        print(distance)

        rag_chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough(), "distance": RunnablePassthrough()} 
            | prompt
            | llm
            | StrOutputParser()
        )

        # Passa 'context' e 'question' para o invoke
        resposta = rag_chain.invoke({"context": context, "question": question})

        if distance > 0.8:
            # Passa 'context' e 'question' para o invoke
            resposta += "\n\nA resposta fornecida é baseada nas informações disponíveis e pode não estar 100% precisa. Recomendo confirmar com profissionais de saúde para informações totalmente confiáveis."

    return resposta