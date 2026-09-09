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

PERSISTENCE_DIR = "./pipeline_storage"
CHROMA_DIR = "./chroma_db"

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

def main() -> None:
    print("Hello from llamaindex-basics!")
   
    documents = SimpleDirectoryReader(
        input_dir="./Files",
        recursive=False,
        required_exts=[".md", ".pdf"],
        num_files_limit=3
    ).load_data()

    print(f"Loaded {len(documents)} documents")

    #Create persistent Chroma vector store
    print("Setting up ChromaDB vector store...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    chroma_collection = chroma_client.get_or_create_collection("llamaindex_docs")
    vectore_store = ChromaVectorStore(chroma_collection=chroma_collection)

    #Check how many docs already in vectore store
    existing_count = chroma_collection.count()
    print(f"ChromaDB already contains {existing_count} embeddings.")

    #If we already have embeddings, skip ingestion and go straight to querying
    if existing_count > 0:
        print("Using existing embeddings from ChromaDB (skipping ingestion).")
    else:
        print("Create ingestion pipeline...")
        pipeline = IngestionPipeline(
            transformations=get_transformation(),
            docstore=SimpleDocumentStore(),
        )
        #Check if we have persisted cache to load
        if os.path.exists(PERSISTENCE_DIR):
            print("Loading persisted document store")
            pipeline.docstore.from_persist_path(PERSISTENCE_DIR)
            print("Loaded persisted store.")

        print("Running ingestion pipelene...")
        process_nodes = pipeline.run(documents=documents) 
        print(f"Process into {len(process_nodes)} nodes with embeddings.")

        #Persist cache for the next run
        print(f"Persisting document store {PERSISTENCE_DIR}")
        pipeline.docstore.persist(persist_path=PERSISTENCE_DIR)

        if process_nodes[0].embedding:
            print(f"Embedding dimensions: {len(process_nodes[0].embedding)}")

        #Extract and print metadata from first node
        first_node_metadata = process_nodes[0].metadata
        for key, value in first_node_metadata.items():
            print(f"  {key} : {value}")

    index = VectorStoreIndex(nodes=process_nodes)
    #query index
    query_engine = index.as_query_engine()
    response = query_engine.query("Give me the names of all candidates")
    print(response)  
    #Create index from the vectore store
    print("Creating Vectore store index from ChromaDB.")
    vectore_index = VectorStoreIndex.from_vector_store(vector_store=vectore_store)
    print("Vectore store index created.")
    query_engine_from_vectore_store = vectore_index.as_query_engine()

    print("\n Query using vector engine")
    vectore_response = query_engine_from_vectore_store.query("Give me the name of HRs")
    print("Query: Give me the name of HRs")
    print(f"Response:\n {vectore_response}")


"""
    node_parser = SentenceSplitter(
        chunk_size=Settings.chunk_size,
        chunk_overlap=Settings.chunk_overlap        
    )

    print("Parse documents into nodes with custom chunking")
    nodes = node_parser.get_nodes_from_documents(documents)
    print(f"Parsed {len(nodes)} nodes from documents")

    #Inspect a few sample nodes
    print(f"Sample nodes after custom chunking: ")
    for i, node in enumerate(nodes[:3]):
        print(f"\n Node {i+1} content: \n {node.get_content()}\n")

    #create index
    #index = VectorStoreIndex.from_documents(
    #    documents
    #)
"""


if __name__ == "__main__":
    main()
