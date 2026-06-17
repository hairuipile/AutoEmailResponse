"""向量索引管理模块：索引构建、清理、检索

本模块负责：
1. 构建向量索引（每个 chunk 包含 parent_uuid 等元信息）
2. 支持按 parent_uuid 批量删除孤立块（源文章被删时调用）
3. 增量入库 + 去重（基于 parent_uuid 避免重复入库）
4. 原子替换某个父文档的所有子块（内容变更时使用）
"""
from pathlib import Path
from typing import Any, Iterable

from langchain_chroma import Chroma

from .manifest import get_deleted_articles
from .vectorstore import get_embeddings, get_vectorstore

PROJECT_ROOT = Path(__file__).parent.parent.parent
COLLECTION_NAME = "langchain"


class VectorIndexer:
    """向量索引管理器"""

    def __init__(self, persist_directory: str = "db"):
        self.persist_directory = persist_directory
        self.db_path = PROJECT_ROOT / persist_directory

    # ---------- 内部辅助 ----------

    def _get_client(self, collection_name: str = COLLECTION_NAME) -> Chroma:
        """获取 Chroma 客户端（统一入口，避免重复连接）"""
        self.db_path.mkdir(parents=True, exist_ok=True)
        return Chroma(
            persist_directory=str(self.db_path),
            embedding_function=get_embeddings(),
            collection_name=collection_name,
        )

    def _get_existing_parent_uuids(self, collection_name: str = COLLECTION_NAME) -> set[str]:
        """查询向量库中已存在的 parent_uuid 集合

        Returns:
            集合中所有不重复的 parent_uuid
        """
        try:
            client = self._get_client(collection_name)
            results = client._collection.get(include=["metadatas"])
            return {
                meta["parent_uuid"]
                for meta in (results.get("metadatas") or [])
                if meta and "parent_uuid" in meta
            }
        except Exception as e:
            print(f"[!] 查询已存在 parent_uuid 失败: {e}")
            return set()

    def _dedupe_by_parent_uuid(
        self,
        chunks: list[Any],
        collection_name: str = COLLECTION_NAME,
    ) -> list[Any]:
        """按 parent_uuid 去重：过滤掉向量库中已存在的父文档对应的所有 chunk

        逻辑：
        - 把所有 chunk 按 parent_uuid 分组
        - 查询向量库中已有的 parent_uuid
        - 只保留「不在向量库」的父文档对应的 chunk

        Args:
            chunks: 待入库的 chunk 列表
            collection_name: 集合名

        Returns:
            去重后的 chunk 列表
        """
        existing = self._get_existing_parent_uuids(collection_name)
        if not existing:
            return chunks

        filtered = [c for c in chunks if c.metadata.get("parent_uuid") not in existing]
        skipped = len(chunks) - len(filtered)
        if skipped:
            print(
                f"[=] 去重过滤：跳过 {skipped} 个 chunk "
                f"（{len(existing & {c.metadata.get('parent_uuid') for c in chunks})} 个父文档已存在）"
            )
        return filtered

    # ---------- 索引构建 ----------

    def build_index(
        self,
        chunks: list[Any],
        collection_name: str = COLLECTION_NAME,
        embeddings=None,
        dedupe: bool = True,
    ) -> Chroma:
        """从文档块构建向量索引（首建/全量重建时使用）

        Args:
            chunks: LangChain Document 对象列表，每个 chunk 应包含 chunk_id 和 parent_uuid
            collection_name: Chroma 集合名称
            embeddings: 嵌入模型（可选）
            dedupe: 是否按 parent_uuid 去重，默认 True

        Returns:
            Chroma 向量库实例
        """
        if not chunks:
            raise ValueError("没有文档块可索引")

        if embeddings is None:
            embeddings = get_embeddings()

        # 去重：跳过向量库中已存在的父文档
        if dedupe:
            chunks = self._dedupe_by_parent_uuid(chunks, collection_name)
            if not chunks:
                print("[=] 没有新增 chunk，跳过 build_index")
                return self._get_client(collection_name)

        self.db_path.mkdir(parents=True, exist_ok=True)

        client = self._get_client(collection_name)
        BATCH_SIZE = 64
        total = 0

        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            ids = [chunk.metadata.get("chunk_id") for chunk in batch]
            client.add_documents(
                documents=batch,
                embedding=embeddings,
                ids=ids,
            )
            total += len(batch)

        print(f"[+] 向量索引已构建，共 {total} 个文档块")
        return client

    def add_to_index(
        self,
        chunks: list[Any],
        collection_name: str = COLLECTION_NAME,
        embeddings=None,
        dedupe: bool = True,
        batch_size: int = 64,
    ) -> int:
        """增量向已有索引追加 chunk（不重建），支持智谱 64 条限制自动分批

        与 build_index 的区别：不会创建新集合，只往现有集合里 add
        天然幂等：基于 chunk_id 主键，重复 add 不会产生重复

        Args:
            chunks: 待追加的 chunk 列表
            collection_name: 集合名
            embeddings: 嵌入模型（可选）
            dedupe: 是否按 parent_uuid 去重
            batch_size: 每次 embed 的最大条数，默认 64（智谱限制）

        Returns:
            实际新增的 chunk 数量
        """
        if not chunks:
            print("[=] 无 chunk 可追加")
            return 0

        if dedupe:
            chunks = self._dedupe_by_parent_uuid(chunks, collection_name)
            if not chunks:
                print("[=] 全部父文档已存在，跳过追加")
                return 0

        if embeddings is None:
            embeddings = get_embeddings()

        client = self._get_client(collection_name)
        total_added = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            ids = [chunk.metadata.get("chunk_id") for chunk in batch]
            client.add_documents(
                documents=batch,
                embedding=embeddings,
                ids=ids,
            )
            total_added += len(batch)

        print(f"[+] 增量追加 {total_added} 个 chunk 到集合 {collection_name}")
        return total_added

    def replace_parent(
        self,
        parent_uuid: str,
        new_chunks: list[Any],
        collection_name: str = COLLECTION_NAME,
        embeddings=None,
    ) -> int:
        """原子替换某个父文档的所有子块（用于内容修改场景）

        流程：先删旧 → 再加新
        注意：Chroma 的 delete + add 不在一个事务里，有极小的窗口期
        生产环境建议用 upsert 替代，这里通过显式删除+add 实现

        Args:
            parent_uuid: 父文档 UUID
            new_chunks: 新切分出的 chunk 列表（metadata 中必须带 parent_uuid）
            collection_name: 集合名
            embeddings: 嵌入模型（可选）

        Returns:
            新增的 chunk 数量
        """
        if embeddings is None:
            embeddings = get_embeddings()

        # 1. 先删旧
        old_count = self.delete_by_parent_uuid(parent_uuid, collection_name)

        # 2. 校验新 chunk 的 parent_uuid 一致性
        for chunk in new_chunks:
            chunk.metadata["parent_uuid"] = parent_uuid

        # 3. 加新（分批，避免智谱单次 64 条限制）
        if new_chunks:
            client = self._get_client(collection_name)
            BATCH_SIZE = 64
            for i in range(0, len(new_chunks), BATCH_SIZE):
                batch = new_chunks[i : i + BATCH_SIZE]
                ids = [chunk.metadata.get("chunk_id") for chunk in batch]
                client.add_documents(
                    documents=batch,
                    embedding=embeddings,
                    ids=ids,
                )

        print(
            f"[+] 替换完成：父文档 {parent_uuid[:8]}... "
            f"删除 {old_count} 个旧 chunk，新增 {len(new_chunks)} 个新 chunk"
        )
        return len(new_chunks)

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
                collection_name=collection_name,
            )
            client.delete_collection()
            print(f"[+] 已清空集合: {collection_name}")
        except Exception as e:
            print(f"[!] 清空集合时出错: {e}")

    def delete_by_parent_uuid(self, parent_uuid: str, collection_name: str = COLLECTION_NAME) -> int:
        """按父文章 UUID 删除所有关联的子块

        Args:
            parent_uuid: 父文章的 UUID
            collection_name: 集合名称

        Returns:
            删除的块数量
        """
        try:
            client = self._get_client(collection_name)
            results = client._collection.get(
                where={"parent_uuid": parent_uuid},
                include=[]
            )
            ids_to_delete = results.get("ids", [])
            if ids_to_delete:
                client._collection.delete(ids=ids_to_delete)
                print(f"[+] 删除了 {len(ids_to_delete)} 个子块 (parent_uuid: {parent_uuid})")
                return len(ids_to_delete)
            return 0
        except Exception as e:
            print(f"[!] 删除块时出错: {e}")
            return 0

    def sync_deleted_articles(self) -> int:
        """同步被删除的源文章：清理其在向量库中的块"""
        deleted = get_deleted_articles()
        if not deleted:
            return 0

        total_removed = 0
        for entry in deleted:
            print(f"[+] 检测到源文件被删除: {entry['file_path']}")
            total_removed += self.delete_by_parent_uuid(entry["uuid"])
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
