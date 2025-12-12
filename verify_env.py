try:
    import streamlit
    import langchain_community
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.llms import Ollama
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    print("Imports successful!")
except ImportError as e:
    print(f"Import failed: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
