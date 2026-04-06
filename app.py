import streamlit as st
import logging
from rag_system import RAGSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="RAG System Assignment", layout="wide")

@st.cache_resource
def initialize_rag_system():
    try:
        return RAGSystem()
    except Exception as e:
        st.error(f"Failed to initialize RAG System: {e}")
        logger.error(f"Initialization error: {e}")
        return None

rag_system = initialize_rag_system()

st.title("Retrieval-Augmented Generation (RAG) System")
st.markdown("A demonstration of a RAG pipeline with a Vector Database, Retrieval, Generation, and a Self-Learning layer.")

if rag_system:
    query = st.text_input("Enter your query:", placeholder="e.g., Where did coffee drinking originate?")
    

    if query:
        with st.spinner("Processing query..."):
            try:

                response, boosted = rag_system.query(query)
                
                st.subheader("Generated Answer")
                st.info(response.response)
                
                if boosted:
                    st.success("🧠 **Self-Learning Triggered:** The system recognized a similar past query and re-ranked the documents based on historical success!")

                st.subheader("Retrieval Details")
                
                if hasattr(response, "source_nodes") and response.source_nodes:
                    retrieved_nodes = response.source_nodes
                    st.markdown(f"**Top {len(retrieved_nodes)} Document Chunks Retrieved:**")
                    
                    for i, node in enumerate(retrieved_nodes):
                        st.text_area(
                            f"Chunk {i+1} (Similarity Score: {node.score:.4f})",
                            node.node.text,
                            height=150,
                            key=f"node_{i}"
                        )
                else:
                    st.warning("No relevant documents were retrieved.")

            except Exception as e:
                st.error(f"An error occurred during query processing: {e}")
                logger.error(f"Query processing error: {e}")

else:
    st.error("RAG System is not available. Please check the logs for initialization errors.")