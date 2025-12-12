#Local RAG: Chat with your data offline (PDF, Excel, CSV)

This project is a Retrieval-Augmented Generation (RAG) application that allows users to chat with their documents offline using local LLMs.

##Features

Multi-Format: PDF, Excel and CSV support.
Local and Private: Runs on Ollama (Llama 3.2).
Smart Memory: Remembers conversation context.
Source Citiation: Shows exactly where the info comes from.

#Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/rag-project.git
cd rag-project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Ollama:
```bash
# Download and install Ollama from: https://ollama.com/
# Add Ollama to your PATH
# Download Llama 3.2 model: ollama serve llama3.2:3b
```

4. Run the app:
```bash
streamlit run app.py
```

#Usage

1. Upload a PDF, Excel, or CSV file.
2. Adjust the temperature (creativity) slider.
3. Ask questions and get answers with citations.

#Customization

- Modify the prompt template in `app.py` to change the conversation style.
- Adjust chunk size and overlap in `app.py` for better document processing.
- Experiment with different LLMs in `app.py`.