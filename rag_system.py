import streamlit as st
import os
import json
import logging
import chromadb
from difflib import SequenceMatcher
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = "./data"
CHROMA_PATH = "./chroma_db"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
GEMINI_MODEL_NAME = "models/gemini-2.5-flash" 
GEMINI_API_KEY = "PUT YOUR GEMINI API KEY"


class RAGSystem:
    def __init__(self):
        try:
            print("\n[INIT] Starting RAG System Initialization...")
            
            self.history_file = "rag_history.json"
            
            self._setup_settings()
            
            self.index = self._get_index()
            
            self.history = self._load_history()
            print(f"[INIT] Loaded {len(self.history)} past interactions from history.")

            if self.index is None:
                if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
                with open(os.path.join(DATA_DIR, "placeholder.txt"), "w") as f:
                    f.write("This is a placeholder document about AI Agents.")
                documents = SimpleDirectoryReader(DATA_DIR).load_data()
                self.index = VectorStoreIndex.from_documents(documents)

            self.query_engine = self.index.as_query_engine(
                similarity_top_k=5, 
                response_mode="compact"
            )
            print("[INIT] System Ready.\n")

        except Exception as e:
            logger.error(f"Failed to initialize RAG System: {e}")
            raise

    def _setup_settings(self):
        self.llm = Gemini(model=GEMINI_MODEL_NAME, api_key=GEMINI_API_KEY, temperature=0.1)
        Settings.llm = self.llm
        Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
        Settings.chunk_size = 512
        Settings.chunk_overlap = 50

    def _get_index(self):
        if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
        db = chromadb.PersistentClient(path=CHROMA_PATH)
        chroma_collection = db.get_or_create_collection("rag_collection")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        if not os.listdir(DATA_DIR): return None
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        return VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    def _load_history(self):
            if os.path.exists(self.history_file):
                try:
                    with open(self.history_file, "r") as f:
                        return json.load(f)
                except json.JSONDecodeError:
                    return []
            return []

    def _save_history(self):
        with open(self.history_file, "w") as f:
            json.dump(self.history, f, indent=4)

    def _find_similar_past_query(self, current_query):
        best_match = None
        highest_ratio = 0.0
        for entry in self.history:
            ratio = SequenceMatcher(None, current_query.lower(), entry["query"].lower()).ratio()
            if ratio > 0.8 and ratio > highest_ratio: 
                highest_ratio = ratio
                best_match = entry
        return best_match

    def query(self, query_str: str):
        response = self.query_engine.query(query_str)
        source_nodes = response.source_nodes
        
        past_experience = self._find_similar_past_query(query_str)
        
        boosted = False
        if past_experience:
            print(f" 🧠 [SELF-LEARNING] Boost activated for similar query.")
            successful_docs = past_experience["used_docs"]
            for node in source_nodes:
                if node.node.text in successful_docs:
                    node.score = node.score * 1.5 
                    boosted = True
            
            source_nodes.sort(key=lambda x: x.score, reverse=True)
            response.source_nodes = source_nodes

        if source_nodes:
            used_docs_texts = [n.node.text for n in source_nodes[:2]]
            self.history.append({
                "query": query_str,
                "used_docs": used_docs_texts
            })
            self._save_history()
            
        return response, boosted

st.set_page_config(page_title="RAG Agent", layout="wide")
st.title("🤖 RAG System with History-Based Self-Learning")

@st.cache_resource
def get_rag_system():
    return RAGSystem()

try:
    rag = get_rag_system()
    st.success("System Connected.")
except Exception as e:
    st.error(f"Error: {e}")

query = st.chat_input("Ask a question...")

if query:
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        response, boosted = rag.query(query)
        
        st.write(response.response)
        
        if boosted:
             st.success("🧠 **Self-Learning Triggered:** I found a similar past query and boosted the most relevant documents based on history!")
        
        with st.expander("Retrieval Details"):
            for node in response.source_nodes:
                st.markdown(f"**Score:** {node.score:.4f}")
                st.text(node.node.text[:200] + "...")
                st.divider()
