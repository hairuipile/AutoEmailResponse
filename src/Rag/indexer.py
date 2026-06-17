"""向量索引管理模块：索引构建、清理、检索

本模块负责：
1. 构建/重建向量索引
2. 把 chunk 的 chroma_id 回写到 article_chunks 关联表
3. 支持按 chroma_id 批量删除孤立块（源文章被删时调用）
"""
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma

from .manifest import record_article_chunks, get_deleted_files
from .vectorstore import get_embeddings, get_vectorstore

PROJECT_ROOT = Path(__file__).parent.parent.parent
COLLECTION_NAME = "langchain"


class VectorIndexer:
    """向量索引管理器"""

    def __init__(self, persist_directory: str = "db"):
        self.persist_directory = persist_directory
        self.db_path = PROJECT_ROOT / persist_directory

    def build_index(
        self,
        chunks: list[Any],
        collection_name: str = COLLECTION_NAME,
        embeddings=None,
    ) -> Chroma:
        """从文档块构建向量索引，并把 chunk→article 关联写回 manifest

        Args:
            chunks: LangChain Document 对象列表
            collection_name: Chroma 集合名称
            embeddings: 嵌入模型（可选）

        Returns:
            Chroma 向量库实例
        """
        if not chunks:
            raise ValueError("没有文档块可索引")

        if embeddings is None:
            embeddings = get_embeddings()

        self.db_path.mkdir(parents=True, exist_ok=True)

        # 按 source 分组，便于建立文章↔块关联
        source_to_chunks: dict[str, list] = {}
        for chunk in chunks:
            source = chunk.metadata.get("source", "<unknown>")
            rel_source = str(Path(source).relative_to(PROJECT_ROOT)) if Path(source).is_absolute() else source
            source_to_chunks.setdefault(rel_source, []).append(chunk)

        # 构建向量索引
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(self.db_path),
            collection_name=collection_name,
        )

        # 记录每篇文章对应的 chroma_id 列表
        all_ids = vectorstore._collection.get(include=[])["ids"]
        cursor = 0
        for file_path, file_chunks in source_to_chunks.items():
            n = len(file_chunks)
            ids = all_ids[cursor: cursor + n]
            record_article_chunks(file_path, ids)
            cursor += n

        print(f"[+] 向量索引已构建，共 {len(chunks)} 个文档块，{len(source_to_chunks)} 个源文件")
        return vectorstore

    def rebuild_index(
        self,
        chunks: list[Any],
        collection_name: str = COLLECTION_NAME,
    ) -> Chroma:
        """清空并重建索引"""
        self.clear_index(collection_name)
        return self.build_index(chunks, collection_name=collection_name)

    def clear_index(self, collection_name: str = COLLECTION_NAME):
        """清空指定集合的索引"""
        try:
            client = Chroma(
                persist_directory=str(self.db_path),
                embedding_function=get_embeddings(),
            )
            client.delete_collection(collection_name)
            print(f"[+] 已清空集合: {collection_name}")
        except Exception as e:
            print(f"[!] 清空集合时出错: {e}")

    def delete_chunks(self, chroma_ids: list[str], collection_name: str = COLLECTION_NAME) -> int:
        """按 chroma_id 批量删除孤立块"""
        if not chroma_ids:
            return 0
        try:
            client = Chroma(
                persist_directory=str(self.db_path),
                embedding_function=get_embeddings(),
            )
            client._collection.delete(ids=chroma_ids)
            print(f"[+] 删除了 {len(chroma_ids)} 个孤立块")
            return len(chroma_ids)
        except Exception as e:
            print(f"[!] 删除块时出错: {e}")
            return 0

    def sync_deleted_articles(self) -> int:
        """同步被删除的源文章：清理其在向量库中的块"""
        docs_dir = PROJECT_ROOT / "context"
        extensions = {".txt", ".md"}

        current_files = set()
        for fp in docs_dir.rglob("*"):
            if fp.is_file() and fp.suffix in extensions:
                current_files.add(str(fp.relative_to(PROJECT_ROOT)))

        deleted = get_deleted_files(current_files)
        if not deleted:
            return 0

        total_removed = 0
        for entry in deleted:
            print(f"[+] 检测到源文件被删除: {entry['file_path']}")
            if entry["chroma_ids"]:
                total_removed += self.delete_chunks(entry["chroma_ids"])
        return total_removed

    def get_retriever(self, collection_name: str = COLLECTION_NAME, k: int = 4):
        """获取检索器"""
        vectorstore = get_vectorstore(self.persist_directory)
        if vectorstore is None:
            raise RuntimeError(f"向量库不存在: {self.persist_directory}")
        return vectorstore.as_retriever(search_kwargs={"k": k})

    def get_collection_stats(self, collection_name: str = COLLECTION_NAME) -> dict[str, Any]:
        """获取集合统计信息"""
        try:
            vectorstore = get_vectorstore(self.persist_directory)
            if vectorstore is None:
                return {"exists": False, "count": 0}

            count = vectorstore._collection.count()
            return {
                "exists": True,
                "count": count,
                "persist_directory": str(self.db_path),
            }
        except Exception as e:
            return {"exists": False, "count": 0, "error": str(e)}
