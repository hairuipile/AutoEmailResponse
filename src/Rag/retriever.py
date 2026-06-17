"""检索模块：基于向量库的相似性搜索"""
from typing import Any

from src.Rag.vectorstore import get_vectorstore


def similarity_search(
    query: str,
    k: int = 4,
    filter_dict: dict | None = None,
    persist_directory: str = "db",
) -> list[dict[str, Any]]:
    """执行相似性搜索

    Args:
        query: 查询文本
        k: 返回的最相似文档数量
        filter_dict: 元数据过滤条件
        persist_directory: 向量库路径

    Returns:
        包含 page_content 和 metadata 的文档列表
    """
    vectorstore = get_vectorstore(persist_directory)
    if vectorstore is None:
        return []

    docs = vectorstore.similarity_search(
        query,
        k=k,
        filter=filter_dict,
    )

    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
        }
        for doc in docs
    ]


def format_search_results(results: list[dict[str, Any]]) -> str:
    """格式化检索结果为可读文本"""
    if not results:
        return ""

    formatted = []
    for i, result in enumerate(results, 1):
        source = result["metadata"].get("source", "未知来源")
        content = result["content"].strip()
        formatted.append(f"[文档 {i}] 来源: {source}\n{content}")

    return "\n\n---\n\n".join(formatted)
