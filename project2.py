
import pandas as pd
import numpy as np
with open("aloobhaat.txt",encoding="utf-8") as f:
    text=f.read()
print(text)

#Data /Doc, loading
from langchain_community.document_loaders import TextLoader
loadder = TextLoader("aloobhaat.txt",encoding ="utf-8")
text_loadder=loadder.load()
print(text_loadder)

from langchain_text_splitters import RecursiveCharacterTextSplitter
textSplitter = RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=20)
finalDoc=textSplitter.split_documents(text_loadder)
print(finalDoc)

from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

from langchain_community.vectorstores import FAISS
db = FAISS.from_documents(finalDoc, embeddings)
db

query = input("Teri gend mein nunu!! Sawal puch!!!")
docs=db.similarity_search(query)
for i in range(len(docs)):
  print(docs[i].page_content)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

# Embedding model
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# Read PDF
loader = PyPDFLoader("aloobhaat.pdf")
documents = loader.load()

# Split PDF into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

finalDoc = text_splitter.split_documents(documents)

# Create FAISS vector database
db = FAISS.from_documents(finalDoc, embeddings)

# Ask question
query = input("Teri gend mein nunu!! Sawal puch!!! ")

# Search relevant PDF chunks
docs = db.similarity_search(query)

# Display results
for i, doc in enumerate(docs):
    page = doc.metadata.get("page", "Unknown")

    print(f"\n--- Page {page + 1} ---")
    print(doc.page_content)