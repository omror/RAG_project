import streamlit as st
import tempfile
import os
import pandas as pd

from langchain_community.document_loaders import PyPDFLoader, CSVLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import Document


st.title("RAG Project")

#Settings
st.sidebar.header("Settings")
selected_model = st.sidebar.text_input("Model Name", value="llama3.2:3b")

#Temperature
temperature = st.sidebar.slider("Creativity (Temperature)", min_value=0.0, max_value=1.0, value=0.1, step=0.1)


#Functions
def create_vector_db(uploaded_file):
    """Reads PDF, CSV, or Excel and creates a vector database."""

    #File extension
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        documents = []

        #PDF
        if file_extension == ".pdf":
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()

        #CSV
        elif file_extension == ".csv":
            loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8")
            documents = loader.load()

        #Excel (.xlsx)
        elif file_extension in [".xlsx", ".xls"]:
            df = pd.read_excel(tmp_file_path)
            for index,row in df.iterrows():
                #Change every row to text
                content = "\n".join([f"{col}: {val}" for col, val in row.items()])
                documents.append(Document(page_content=content, metadata={"source": f"Row {index+1}"}))

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)

        embeddings = OllamaEmbeddings(model = selected_model)
        vectorstore = FAISS.from_documents(documents = splits, embedding = embeddings)
        return vectorstore
    
    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)


def get_conversation_chain(vectorstore, model_name, temp_value):
    llm = Ollama(model=model_name, temperature=temp_value)

    #Memory object: Keeps the conversation in mind
    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True,
        output_key="answer"
        )


    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vectorstore.as_retriever(),
        memory = memory,
        return_source_documents=True
    )

    return conversational_chain

#Main Flow (With Session State)

# Initialize session state for vectorstore if it doesn't exist.
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader("Upload file (PDF, Excel, CSV)", type = ["pdf", "xlsx", "csv"])

if uploaded_file is not None:
    if st.session_state.vectorstore is None or st.session_state.conversation is None:
        with st.spinner("Processing file..."):
            st.session_state.vectorstore = create_vector_db (uploaded_file)

            #Temperature value
            st.session_state.conversation = get_conversation_chain(
                st.session_state.vectorstore, selected_model, temperature
            )
            st.success("File processed. Chat started.")

    #Chat History
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    #New Question
    if prompt := st.chat_input("Ask a question"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if st.session_state.conversation:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.conversation({"question": prompt})
                    answer = response["answer"]
                    source_docs = response["source_documents"]

                    #Showing Documents
                    source_text = ""

                    if source_docs:
                        source_text = "\n\n--- \n** Sources:**\n"
                        for i, doc in enumerate(source_docs[:3]): #First 3 documents
                            loc = doc.metadata.get("page", doc.metadata.get("source", "unknown"))
                            content_preview = doc.page_content[:100].replace("\n", " ")
                            source_text += f"**{i+1}. [Loc: {loc}]** *...{content_preview}...*\n"

                    final_answer = answer + source_text
                    st.markdown(final_answer)

            st.session_state.chat_history.append({"role": "assistant", "content": final_answer})

        
    




