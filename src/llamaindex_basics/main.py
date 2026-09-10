import os
import time
import chromadb
import streamlit as st
from dotenv import load_dotenv

# Core LlamaIndex imports
from llama_index.core import Document, Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.extractors import KeywordExtractor, SummaryExtractor, TitleExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.readers.file import PyMuPDFReader
from llama_index.vector_stores.chroma import ChromaVectorStore

# Import your agent constructor
from src.llamaindex_basics.tools import get_agent

CACHE_DIR = "./pipeline_cache"
CHROMA_DIR = "./chroma_db_cached"

load_dotenv()

# Global settings
Settings.llm = OpenAI(model="gpt-4o-mini")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.chunk_size = 512
Settings.chunk_overlap = 50


def get_transformation():
    return [
        SentenceSplitter(
            chunk_size=Settings.chunk_size,
            chunk_overlap=Settings.chunk_overlap,
        ),
        TitleExtractor(),
        SummaryExtractor(),
        KeywordExtractor(),
        OpenAIEmbedding(model=Settings.embed_model.model_name),
    ]


def get_index():
    # Use PyMuPDFReader to correctly parse PDF resumes without character encoding errors
    file_extractor = {".pdf": PyMuPDFReader()}

    documents = SimpleDirectoryReader(
        input_dir="./Files",
        recursive=False,
        required_exts=[".md", ".pdf"],
        file_extractor=file_extractor,
        num_files_limit=50,  # Increased limit to ensure all candidate files are read
    ).load_data()
    print(f"Loaded {len(documents)} documents")

    # Setup Chroma Vector Store
    print("[2/6] Setting up ChromaDB vector store...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    chroma_collection = chroma_client.get_or_create_collection("llamaindex_docs")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    print(f"Chroma path: {CHROMA_DIR}")
    print(f"Existing embeddings in ChromaDB: {chroma_collection.count()}")

    # Setup Ingestion Pipeline
    print("[3/6] Creating ingestion pipeline with caching...")
    pipeline = IngestionPipeline(
        transformations=get_transformation(),
        vector_store=vector_store,
        docstore=SimpleDocumentStore(),  # Tracks document hashes
    )

    if os.path.exists(CACHE_DIR):
        print(f"Loading existing cache from {CACHE_DIR}...")
        pipeline.load(persist_dir=CACHE_DIR)
        print("Cache loaded. Unchanged documents will be skipped.")
    else:
        print("No existing cache found. Will process all documents.")

    # Run Ingestion Pipeline
    print("\n[4/6] Running ingestion pipeline...")
    start_time = time.time()

    process_nodes = pipeline.run(
        documents=documents, num_workers=4, show_progress=True
    )

    elapsed = time.time() - start_time
    print(f"\nPipeline completed in {elapsed:.2f} seconds.")
    print(f"Nodes returned: {len(process_nodes)}")
    print(f"Total embeddings in ChromaDB: {chroma_collection.count()}")

    # Persist cache
    print(f"[5/6] Persisting cache to {CACHE_DIR}")
    pipeline.persist(CACHE_DIR)

    print("\n[6/6] Creating vector store index...")
    vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    return vector_index


def main():
    st.set_page_config(
        page_title="Talent Search AI", layout="wide", page_icon="📄"
    )
    st.title("Talent Search AI")
    st.caption("Ask questions about candidate resumes, general HR practices, or web trends.")

    # Ensure documents are ingested into ChromaDB
    get_index()

    # Initialize ReAct Agent in Streamlit session state
    if "agent" not in st.session_state:
        st.session_state.agent = get_agent()

    # Initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat interaction
    if prompt := st.chat_input("Ask a question about resumes or candidates..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing tools and reasoning..."):
                # Execute user query through ReAct Agent
                response = st.session_state.agent.chat(prompt)
                st.session_state.messages.append(
                    {"role": "assistant", "content": str(response)}
                )
            st.markdown(str(response))


if __name__ == "__main__":
    main()