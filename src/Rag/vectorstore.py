"""向量库管理模块"""
import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.embeddings import ZhipuAIEmbeddings

PROJECT_ROOT = Path(__file__).parent.parent.parent


def get_embeddings():
    """获取嵌入模型"""
    zhipuai_api_key = os.getenv("ZHIPUAI_API_KEY", "")
    if not zhipuai_api_key:
        raise ValueError("ZHIPUAI_API_KEY 环境变量未设置")
    return ZhipuAIEmbeddings(model="embedding-3", api_key=zhipuai_api_key)


def get_vectorstore(persist_directory: str = "db"):
    """获取已存在的向量库"""
    db_path = PROJECT_ROOT / persist_directory
    if not db_path.exists():
        return None
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=str(db_path),
        embedding_function=embeddings,
    )
