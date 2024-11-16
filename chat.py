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

def get_contexts(pergunta):
    results = vector_store.similarity_search_with_score(pergunta)

    # Criar um novo array com os page_content e score menor 1
    relevantDocuments = [(document.page_content, score) for document, score in results if score < 1]

    # Exibir o novo array
    # if len(relevantDocuments) == 0:
    #     print("relevantDocuments está vazio")
    # else:
    #     for i, (page_content, score) in enumerate(relevantDocuments, start=1):
    #         print(f"Item {i}:")
    #         print(f"  Page Content: {page_content}")
    #         print(f"  Score: {score}")

    print(f"Tamanho relevantDocuments {len(relevantDocuments)}")

    return relevantDocuments

def process_question(question):

    contexts = get_contexts(question)
    print(contexts)

    resposta = ""

    if len(contexts) == 0:
        resposta = "Parece que essa pergunta está fora do meu tema principal, que é amamentação. Se precisar de informações ou apoio sobre amamentação, estou aqui para ajudar no que for possível!"

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

        rag_chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough(), "distance": RunnablePassthrough()} 
            | prompt
            | llm
            | StrOutputParser()
        )

        #Passa 'context' e 'question' para o invoke
        resposta = rag_chain.invoke({"context": contexts, "question": question})

        if len(contexts) < 4:
            resposta += "\n\nA resposta fornecida é baseada nas informações disponíveis e pode não estar 100% precisa. Recomendo confirmar com profissionais de saúde para informações totalmente confiáveis."

    return resposta