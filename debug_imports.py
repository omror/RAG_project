print("Importing PyPDFLoader...")
from langchain_community.document_loaders import PyPDFLoader
print("Importing RecursiveCharacterTextSplitter...")
from langchain_text_splitters import RecursiveCharacterTextSplitter
print("Importing OllamaEmbeddings...")
from langchain_community.embeddings import OllamaEmbeddings
print("Importing FAISS...")
from langchain_community.vectorstores import FAISS
print("Importing Ollama...")
from langchain_community.llms import Ollama
print("Importing RetrievalQA...")
from langchain.chains import RetrievalQA
print("Importing PromptTemplate...")
from langchain.prompts import PromptTemplate
print("All imports successful.")
