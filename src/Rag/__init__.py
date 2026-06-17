"""RAG 模块：知识库加载、分块、向量化、检索"""
from src.Rag.manifest import (
    check_knowledge_changes,
    save_manifest,
)
from src.Rag.loader import (
    get_documents,
    split_documents,
)
from src.Rag.vectorstore import (
    get_embeddings,
    get_vectorstore,
)
from src.Rag.indexer import (
    VectorIndexer,
)
from src.Rag.retriever import (
    similarity_search,
    format_search_results,
)

__all__ = [
    # manifest
    "check_knowledge_changes",
    "save_manifest",
    # loader
    "get_documents",
    "split_documents",
    # vectorstore
    "get_embeddings",
    "get_vectorstore",
    # indexer
    "VectorIndexer",
    # retriever
    "similarity_search",
    "format_search_results",
]
