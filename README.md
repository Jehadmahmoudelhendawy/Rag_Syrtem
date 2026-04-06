# Retrieval-Augmented Generation (RAG) System

This project is a complete implementation of a RAG system, featuring a Vector Database (ChromaDB), a Retrieval Mechanism (using BGE Embeddings), and a Generation Module (using an LLM). It also includes a friendly User Interface (Streamlit) and a Self-Learning/Self-Correction layer.

## Prerequisites

You need to have Python 3.8+ and `pip` installed.

## Setup and Execution

### 1. Install Dependencies

Open your terminal and run the following command to install all required Python libraries:

```bash
pip install llama-index llama-index-vector-stores-chroma sentence-transformers openai pydantic streamlit
```

### 2. Set Up API Key

The system requires an API key to access the Large Language Model (LLM).

1.  **Obtain Key:** You must use a Manus API key, as the system is configured to use the Manus API endpoint.
2.  **Set Environment Variable:** Set the API key as an environment variable.

**For Linux/macOS:**
```bash
export MANUS_API_KEY="YOUR_API_KEY"
```

**For Windows (Command Prompt):**
```bash
set MANUS_API_KEY="YOUR_API_KEY"
```

### 3. Project Structure

Ensure your file structure looks like this:

```
rag_system/
├── app.py
├── rag_system.py
├── data/
│   └── documents.txt
└── README.md
```

The `documents.txt` file contains the source documents used to build the vector database.

### 4. Run the System

**First: Build the Vector Database**

When you run the UI for the first time, the system will automatically build the ChromaDB vector database in a local folder named `chroma_db/`.

**Second: Run the User Interface (UI)**

Navigate to the `rag_system` directory in your terminal and run the Streamlit application:

```bash
cd rag_system
streamlit run app.py
```

The application will automatically open in your browser at `http://localhost:8501`.

