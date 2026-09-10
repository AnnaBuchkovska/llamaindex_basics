import os
import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import ReActAgent
from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec

CHROMA_DIR = "./chroma_db_cached"


# ---------------------------------------------------------
# TASK 1: Implement the Retrieval Tool
# ---------------------------------------------------------
def create_resume_retrieval_tool() -> QueryEngineTool:
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    chroma_collection = chroma_client.get_or_create_collection("llamaindex_docs")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    query_engine = vector_index.as_query_engine(
        similarity_top_k=5,
        response_mode="compact"
    )

    return QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="resume_search_tool",
            description=(
                "Useful for searching and retrieving candidate information from the resume database. "
                "Use this tool when answering questions about candidates' skills, experience, education, "
                "work history, or when listing candidates matching specific criteria."
            )
        )
    )


# ---------------------------------------------------------
# TASK 2: Develop Additional Tools
# ---------------------------------------------------------
def create_general_knowledge_tool() -> FunctionTool:
    llm = OpenAI(model="gpt-4o-mini")

    def ask_general_knowledge(query: str) -> str:
        response = llm.complete(query)
        return str(response)

    return FunctionTool.from_defaults(
        fn=ask_general_knowledge,
        name="general_knowledge_tool",
        description=(
            "Useful for answering general knowledge questions, HR best practices, technology definitions, "
            "or any query completely unrelated to candidate resumes."
        )
    )


def create_web_search_tool() -> list:
    tool_spec = DuckDuckGoSearchToolSpec()
    return tool_spec.to_tool_list()


# ---------------------------------------------------------
# TASK 3: Utilize Function / ReAct Agent
# ---------------------------------------------------------
def get_agent() -> ReActAgent:
    toolkit = [
        create_resume_retrieval_tool(),
        create_general_knowledge_tool()
    ] + create_web_search_tool()

    llm = OpenAI(model="gpt-4o-mini", temperature=0.1)

    system_prompt = (
        "You are an AI HR Assistant specializing in candidate resume analysis. "
        "Use your toolkit to answer user questions efficiently."
    )

    agent = ReActAgent.from_tools(
        tools=toolkit,
        llm=llm,
        system_prompt=system_prompt,
        verbose=True
    )

    return agent