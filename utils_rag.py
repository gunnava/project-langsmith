"""
RAG utilities.
Builds (or loads from cache) a SKLearnVectorStore over LangSmith docs,
and exposes a LangChain @tool for use in the agent.
"""

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders.sitemap import SitemapLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool

_PERSIST_PATH = Path(__file__).parent / "langsmith_docs.parquet"


def get_vector_db_retriever():
    """Return a retriever backed by a persisted SKLearnVectorStore of LangSmith docs."""
    embd = OpenAIEmbeddings()

    if _PERSIST_PATH.exists():
        vectorstore = SKLearnVectorStore(
            embedding=embd,
            persist_path=str(_PERSIST_PATH),
            serializer="parquet",
        )
        print(f"[Docs] Loaded vectorstore from {_PERSIST_PATH.name}")
        return vectorstore.as_retriever(search_kwargs={"k": 8})

    print("[Docs] Building vectorstore (one-time, ~7 min)...")

    sitemap_docs = SitemapLoader(
        web_path="https://docs.langchain.com/sitemap.xml",
        filter_urls=["https://docs.langchain.com/langsmith/"],
        continue_on_failure=True,
    ).load()

    # Load reference page directly to double its chunk count in the index
    reference_docs = WebBaseLoader(
        ["https://docs.langchain.com/langsmith/reference"]
    ).load()

    docs = sitemap_docs + reference_docs
    if not docs:
        raise RuntimeError("SitemapLoader returned 0 documents — check the sitemap URL.")
    print(f"[Docs] Loaded {len(docs)} pages total")

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=0
    )
    splits = splitter.split_documents(docs)
    if not splits:
        raise RuntimeError("Document splitting produced 0 chunks.")
    print(f"[Docs] {len(splits)} chunks — embedding now...")

    vectorstore = SKLearnVectorStore.from_documents(
        documents=splits,
        embedding=embd,
        persist_path=str(_PERSIST_PATH),
        serializer="parquet",
    )
    vectorstore.persist()
    print(f"[Docs] Saved to {_PERSIST_PATH.name}")
    return vectorstore.as_retriever(search_kwargs={"k": 8})


retriever = get_vector_db_retriever()


@tool
def search_langsmith_docs(query: str, category: str = "sdk_setup") -> str:
    """Search LangSmith official documentation for articles matching the developer's question.

    Args:
        query: Keywords from the developer's question
        category: Support category — tracing, evaluation, prompt_management,
                  feedback, monitoring, or sdk_setup
    """
    enriched_query = f"{category} {query}"
    docs = retriever.invoke(enriched_query)
    if not docs:
        return "No relevant documentation found."
    return "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )
