from dotenv import load_dotenv
import os

from llama_index.core import SimpleDirectoryReader

#core data structure - docs and settings
from llama_index.core import Document
#general setting of all project
from llama_index.core import Settings

#Text splitters
from llama_index.core.node_parser import SentenceSplitter

#embeddings models
from llama_index.embeddings.openai import OpenAIEmbedding

#Index creation - Vector store index
from llama_index.core import VectorStoreIndex

#LLM configuration  -- OPENAI
from llama_index.llms.openai import OpenAI

#To implement Pipeline architecture
from llama_index.core.ingestion import IngestionPipeline

from llama_index.core.extractors import TitleExtractor, SummaryExtractor, KeywordExtractor

#For caching
from llama_index.core.storage.docstore import SimpleDocumentStore

#For persistent vector store
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

import streamlit as st

from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.chat_engine.types import ChatMode

CACHE_DIR = "./pipeline_cache"
CHROMA_DIR = "./chroma_db_cached"

def get_transformation():
    return [
            SentenceSplitter(chunk_size=Settings.chunk_size,
                             chunk_overlap=Settings.chunk_overlap,
            ),
            TitleExtractor(),
            SummaryExtractor(),
            KeywordExtractor(),
            OpenAIEmbedding(model=Settings.embed_model.model_name),
        ]


load_dotenv()


Settings.llm = OpenAI(model="gpt-4o-mini")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.chunk_size = 512
Settings.chunk_overlap = 50

def get_index():       
        documents = SimpleDirectoryReader(
                input_dir="./Files",
                recursive=False,
                required_exts=[".md", ".pdf"],
                num_files_limit=10
            ).load_data()
        print(f"Loaded {len(documents)} documents")
    
        #Create persistent Chroma vector store
        print(f"[2/6] Setting up ChromaDB vectore store...")
        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        chroma_collection = chroma_client.get_or_create_collection("llamaindex_docs")
        vectore_store = ChromaVectorStore(chroma_collection=chroma_collection)
        print(f"Chroma path: {CHROMA_DIR}")
        print(f"Existing embeddings in ChromaDB: {chroma_collection.count()}")
    
        #Create pipeline docstore for deduplication
        print(f"[3/6] Creating ingestion pipeline with caching...")
        pipeline = IngestionPipeline(
            transformations=get_transformation(),
            vector_store=vectore_store,
            docstore=SimpleDocumentStore() # Track documents hashes
        )
    
        if(os.path.exists(CACHE_DIR)):
            print(f"Loading existing cache from {CACHE_DIR}...")
            pipeline.load(persist_dir=CACHE_DIR)
            print(" Cache loaded. Unchanged documents will be skipped.")
        else:
            print("No existing cache found. Will process all documents.")
    
        #Run the pipeline - LlamaIndex will use cached transformations
        print("\n[4/6] Running ingestion pipeline...")
        print(" Cached transformations will be reused - no redundant API calls.")
    
        import time
        start_time = time.time()
    
        process_nodes = pipeline.run(
            documents=documents,
            num_workers=4,
            show_progress=True
        )
    
        elapsed = time.time() - start_time
    
        #Report results
        print(f"\n Pipeline completed in {elapsed} seconds.")
        print(f"Nodes returned: {len(process_nodes)}")
        print(f" Total embeddings in ChromaDB: {chroma_collection.count()}")
    
        if process_nodes:
            print("\n Sample metadata from first NEW node:")
            if process_nodes[0].embedding:
                print(f"Embedding dimensions: {len(process_nodes[0].embedding)}")
                #Extract and print metadata from first node
                first_node_metadata = process_nodes[0].metadata
                for key, value in first_node_metadata.items():
                    print(f"  {key} : {value}")
        #Persist cache for the next run
        print(f"[5/6] Persisting cache to {CACHE_DIR}")
        pipeline.persist(CACHE_DIR)
        print("Cached saved. Next run will skip unchanged documents.")
    
    
        print("\n [6/6] Creating vectore store index and testing query...")
        vectore_index = VectorStoreIndex.from_vector_store(vector_store=vectore_store)
        return vectore_index


def main():
    st.set_page_config(page_title="RAG with Vectore Store", layout="wide", page_icon="assets/Llama.png")
    st.title("Talent Search AI")
    st.caption("Ask question about resume content")
     
    #Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.message = []
     
    #Initialize chat engine in session state
    if "chat_engine" not in st.session_state:
        index = get_index()
        memory = ChatMemoryBuffer.from_defaults(token_limit=3900)
        st.session_state.chat_engine = index.as_chat_engine(
             memory=memory,
             chat_mode=ChatMode.BEST,
             system_prompt = ("You are a helpful assistant that answers questions about candidates based on the provided resume database. "
                            "Your goal is to extract accurate details regarding skills, experience, education, and work history. "
                            "If the information is not present in the resumes, state that it is not available."
                            )
        )
    #Display chat message from history
    for message in st.session_state.message:
         with st.chat_message(message["role"]):
              st.markdown(message["content"])

    #Chat input
    if prompt := st.chat_input("Ask question any question about resumes..."):
        #Add user message to chat history
        st.session_state.message.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        #Get response from chat engine
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):           
                response = st.session_state.chat_engine.chat(prompt)
                #Update last message with actual response
                st.session_state.message.append(
                    {"role": "assistant", "content": response.response}
                )
            st.markdown(response.response)
     

if __name__ == "__main__":
     main()