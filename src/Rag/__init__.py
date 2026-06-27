"""RAG 模块：知识库加载、分块、向量化、检索"""
import importlib

__all__ = [
    "check_knowledge_changes", "save_manifest", "get_documents", "split_documents",
    "clean_knowledge", "clean_knowledge_idempotent", "get_embeddings", "get_vectorstore",
    "VectorIndexer", "similarity_search", "format_search_results",
]
_LAZY = {
    "check_knowledge_changes": (".manifest", "check_knowledge_changes"),
    "save_manifest": (".manifest", "save_manifest"),
    "get_documents": (".loader", "get_documents"),
    "split_documents": (".loader", "split_documents"),
    "clean_knowledge": (".data_cleaner", "clean_knowledge"),
    "clean_knowledge_idempotent": (".data_cleaner", "clean_knowledge_idempotent"),
    "get_embeddings": (".vectorstore", "get_embeddings"),
    "get_vectorstore": (".vectorstore", "get_vectorstore"),
    "VectorIndexer": (".indexer", "VectorIndexer"),
    "similarity_search": (".retriever", "similarity_search"),
    "format_search_results": (".retriever", "format_search_results"),
}

def __getattr__(name: str):
    if name not in _LAZY: raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod, attr = _LAZY[name]
    return getattr(importlib.import_module(mod, __name__), attr)
