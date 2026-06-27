"""检索模块：多策略融合 + 智能上下文组装

三级检索管线：
1. 向量搜索（Chroma similarity_search）+ BM25 同步并行
2. RRF（Reciprocal Rank Fusion）融合两组排序
3. 候选 child_chunks 按 cosine 余弦重排（query embedding vs chunk embedding）
4. 父子回填：同源 child 关联到 parent，获取完整上下文
5. 智能拼装：贪心按分数往缓冲区塞，总长上限 2800 字
6. 同源去重 / 父子折叠：同 parent_uuid 只返回最相关的 child

调用示例：
    from src.Rag.retriever import advanced_retrieve

    bundle = advanced_retrieve("发票怎么开？", k=8, max_context_chars=2800)
    print(bundle.context_text)      # 拼装好的上下文字符串
    print(bundle.chunks)             # 原始 chunk 列表（带 parent 内容）
    print(bundle.parent_map)         # chunk_id → parent Document
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from langchain_core.documents import Document
from tqdm import tqdm

from .vectorstore import get_embeddings, get_vectorstore

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False


# ---------- 数据结构 ----------


@dataclass
class ContextBundle:
    """检索结果包裹，含拼装上下文 + 原始块 + 来源信息"""

    query: str
    context_text: str                          # 拼装好的可读字符串
    chunks: List[Document]                     # 原始 chunk 列表（已注入 parent 内容）
    parent_map: dict[str, Document]             # chunk_id → parent Document
    sources: List[dict]                        # 下游 answer_generator 用作角标
    reranked_scores: dict[str, float]          # chunk_id → cosine 重排分数

    def __len__(self) -> int:
        return len(self.context_text)

    def is_empty(self) -> bool:
        return len(self.chunks) == 0


# ---------- BM25 辅助 ----------

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9_]+", text.lower())


# ---------- AdvancedRetriever ----------


class AdvancedRetriever:
    """多策略融合 + 智能上下文检索器"""

    def __init__(
        self,
        persist_directory: str = "db",
        embeddings=None,
        alpha: float = 0.6,
        k_vector: int = 20,
        k_bm25: int = 20,
        k_final: int = 8,
        max_context_chars: int = 2800,
        rerank_top_k: int = 12,
    ):
        """
        Args:
            persist_directory: 向量库路径
            embeddings: 嵌入模型（默认从 vectorstore 加载）
            alpha: RRF 中向量搜索的权重系数，beta=1-alpha 为 BM25 权重
            k_vector: 向量搜索取 top_k
            k_bm25: BM25 搜索取 top_k
            k_final: 重排后取 top_k 个进入上下文拼装
            max_context_chars: 拼装缓冲区总字数上限
            rerank_top_k: cosine 重排时参与重排的候选数（需在 k_vector+k_bm25 范围内）
        """
        self.persist_directory = persist_directory
        self.embeddings = embeddings or get_embeddings()
        self.alpha = alpha
        self.k_vector = k_vector
        self.k_bm25 = k_bm25
        self.k_final = k_final
        self.max_context_chars = max_context_chars
        self.rerank_top_k = rerank_top_k

        self._vectorstore = get_vectorstore(persist_directory)
        if self._vectorstore is None:
            raise RuntimeError(f"向量库不存在: {persist_directory}，请先运行预处理")
        self._collection = self._vectorstore._collection
        self._embedding_dim: Optional[int] = None
        self._bm25_corpus: Optional[List[List[str]]] = None
        self._bm25_index: Optional[BM25Okapi] = None
        self._bm25_ids: Optional[List[str]] = None
        self._bm25_built = False

    # ---------- BM25 构建 ----------

    def _ensure_bm25(self) -> None:
        """惰性构建 BM25 索引（只执行一次）"""
        if not _BM25_AVAILABLE:
            return
        if self._bm25_built:
            return

        try:
            results = self._collection.get(include=["documents", "metadatas"])
            texts: List[str] = results.get("documents") or []
            metas: List[dict] = results.get("metadatas") or []
            if not texts:
                self._bm25_built = True
                return
            self._bm25_corpus = [_tokenize(t) for t in texts]
            self._bm25_index = BM25Okapi(self._bm25_corpus)
            self._bm25_ids = [m.get("chunk_id", str(i)) for i, m in enumerate(metas)]
            self._bm25_built = True
        except Exception as e:
            print(f"[!] BM25 索引构建失败: {e}")
            self._bm25_built = True  # 避免重复尝试

    # ---------- 策略 1：向量搜索 + 带分数 ----------

    def _vector_search_with_scores(self, query: str) -> List[Tuple[Document, float]]:
        """返回 [(doc, score)]，score 为 Chroma relevance_score"""
        try:
            results = self._vectorstore.similarity_search_with_relevance_scores(
                query, k=self.k_vector
            )
            return results  # type: ignore[return-value]
        except Exception as e:
            print(f"[!] 向量搜索失败: {e}")
            return []

    # ---------- 策略 2：BM25 ----------

    def _bm25_search(self, query: str) -> List[Tuple[str, float]]:
        """返回 [(chunk_id, score)]，score 为 BM25 原始分数"""
        self._ensure_bm25()
        if self._bm25_index is None or self._bm25_corpus is None or self._bm25_ids is None:
            return []
        tokens = _tokenize(query)
        scores = self._bm25_index.get_scores(tokens)
        scored = sorted(zip(self._bm25_ids, scores), key=lambda x: x[1], reverse=True)
        return scored[: self.k_bm25]

    # ---------- 策略 3：RRF 融合 ----------

    @staticmethod
    def _rrf_fuse(
        vector_ranked: List[Tuple[Document, float]],
        bm25_ranked: List[Tuple[str, float]],
        alpha: float = 0.6,
    ) -> List[Tuple[str, float, Optional[Document]]]:
        """Reciprocal Rank Fusion

        Returns:
            [(chunk_id, rrf_score, doc_or_none), ...]，按 rrf_score 降序
        """
        k = 60  # RRF 稳定性参数

        # 向量排名：doc → rank
        vec_ranks: dict[str, int] = {}
        for rank, (doc, _) in enumerate(vector_ranked, start=1):
            cid = doc.metadata.get("chunk_id", "")
            if cid:
                vec_ranks[cid] = rank

        # BM25 排名：chunk_id → rank
        bm25_ranks: dict[str, int] = {}
        for rank, (cid, _) in enumerate(bm25_ranked, start=1):
            bm25_ranks[cid] = rank

        # 合并 id 集合
        all_ids: set[str] = set(vec_ranks) | set(bm25_ranks)

        # doc 映射
        doc_map: dict[str, Document] = {
            doc.metadata.get("chunk_id", ""): doc
            for doc, _ in vector_ranked
        }

        rrf_scores: dict[str, float] = {}
        for cid in all_ids:
            vec_r = vec_ranks.get(cid, k + 1)
            bm25_r = bm25_ranks.get(cid, k + 1)
            rrf_scores[cid] = alpha / (k + vec_r) + (1 - alpha) / (k + bm25_r)

        fused = sorted(
            [(cid, rrf_scores[cid], doc_map.get(cid)) for cid in all_ids],
            key=lambda x: x[1],
            reverse=True,
        )
        return fused

    # ---------- 策略 4：Cosine 余弦重排 ----------

    def _rerank_by_cosine(
        self, candidates: List[Tuple[str, float, Optional[Document]]], query: str
    ) -> List[Tuple[str, float, Document]]:
        """对候选 chunk 用 query embedding 做 cosine 重排

        流程：
        1. 编码 query embedding
        2. 批量 fetch 候选 chunk 的 embedding
        3. 计算余弦相似度
        4. 按 cosine 分数重排，取 top k_final
        """
        # 取 top rerank_top_k 进入重排
        top_candidates = candidates[: self.rerank_top_k]

        if not top_candidates:
            return []

        # 过滤出有 doc 的候选
        valid: List[Tuple[str, Document]] = [
            (cid, doc) for cid, _, doc in top_candidates if doc is not None
        ]
        if not valid:
            return []

        # 查询向量
        try:
            query_emb = self.embeddings.embed_query(query)
        except Exception as e:
            print(f"[!] Query embedding 失败: {e}，跳过重排")
            return [(cid, orig_score, doc) for cid, orig_score, doc in valid]

        # 批量取 chunk embedding
        chunk_ids = [cid for cid, _ in valid]
        try:
            records = self._collection.get(ids=chunk_ids, include=["embeddings", "documents", "metadatas"])
            emb_raw = records.get("embeddings")
            if emb_raw is None:
                embeddings_list = []
            elif getattr(emb_raw, "ndim", 0) == 2:
                embeddings_list = [emb_raw[i] for i in range(emb_raw.shape[0])]
            else:
                embeddings_list = list(emb_raw)
            docs_list = records.get("documents") if records.get("documents") is not None else []
            metas_list = records.get("metadatas") if records.get("metadatas") is not None else []
        except Exception as e:
            print(f"[!] 批量获取 embeddings 失败: {e}，跳过重排")
            return [(cid, 1.0, doc) for cid, doc in valid]

        # cosine 计算（query_emb 是归一化的，chroma embeddings 也是归一化的）
        cos_scores: dict[str, float] = {}
        for i, emb in enumerate(embeddings_list):
            if emb is None:
                continue
            dot = sum(q * e for q, e in zip(query_emb, emb))
            cid = metas_list[i].get("chunk_id", "") if i < len(metas_list) else ""
            if cid:
                cos_scores[cid] = dot

        # 重排：按 cosine 分数
        reranked = []
        for cid, doc in valid:
            score = cos_scores.get(cid, 0.0)
            reranked.append((cid, score, doc))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[: self.k_final]

    # ---------- 策略 5：父子回填 ----------

    def _parent_fill(
        self, chunks: List[Document]
    ) -> Tuple[List[Document], dict[str, Document]]:
        """把 child chunk 的 parent_uuid 对应的父文档内容回填到 chunk.metadata

        Returns:
            (chunks_with_parent, parent_id → parent_doc map)
        """
        # 收集所有需要的 parent_uuid
        parent_uuids: set[str] = set()
        for chunk in chunks:
            pu = chunk.metadata.get("parent_uuid", "")
            if pu:
                parent_uuids.add(pu)

        if not parent_uuids:
            return chunks, {}

        # 批量查询父文档
        parent_map: dict[str, Document] = {}
        try:
            results = self._collection.get(
                where=[{"parent_uuid": pu} for pu in parent_uuids],
                include=["documents", "metadatas"],
            )
            docs_raw: List[str] = results.get("documents") or []
            metas_raw: List[dict] = results.get("metadatas") or []
            for i, meta in enumerate(metas_raw):
                pu = meta.get("parent_uuid", "")
                doc_type = meta.get("doc_type", "")
                if pu in parent_uuids and doc_type == "parent":
                    parent_map[pu] = Document(
                        page_content=docs_raw[i],
                        metadata=dict(meta),
                    )
        except Exception:
            # 降级：逐个查
            for pu in parent_uuids:
                try:
                    records = self._collection.get(
                        where={"parent_uuid": pu, "doc_type": "parent"},
                        include=["documents", "metadatas"],
                    )
                    docs_raw = records.get("documents") or []
                    metas_raw = records.get("metadatas") or []
                    if docs_raw:
                        parent_map[pu] = Document(
                            page_content=docs_raw[0],
                            metadata=dict(metas_raw[0]),
                        )
                except Exception:
                    pass

        # 把 parent_content 回填到每个 chunk
        for chunk in chunks:
            pu = chunk.metadata.get("parent_uuid", "")
            if pu and pu in parent_map:
                chunk.metadata["parent_content"] = parent_map[pu].page_content

        return chunks, parent_map

    # ---------- 策略 6：同源去重 / 父子折叠 ----------

    @staticmethod
    def _dedup_and_fold(
        reranked: List[Tuple[str, float, Document]]
    ) -> List[Document]:
        """同 parent_uuid 只保留 cosine 分数最高的 child

        Returns:
            去重后的 chunk 列表（保留原始 Document 对象）
        """
        # parent_uuid → (score, doc)
        best_per_parent: dict[str, Tuple[float, Document]] = {}
        for cid, score, doc in reranked:
            pu = doc.metadata.get("parent_uuid", "")
            if pu not in best_per_parent or score > best_per_parent[pu][0]:
                best_per_parent[pu] = (score, doc)

        # 恢复原始 chunk（不要 parent 替换 child）
        seen_ids: set[str] = set()
        unique: List[Document] = []
        for cid, score, doc in reranked:
            if cid in seen_ids:
                continue
            # 只要 best
            if pu := doc.metadata.get("parent_uuid", ""):
                if best_per_parent[pu][1].metadata.get("chunk_id") != cid:
                    continue  # 不是这个 parent 的 best，跳过
            seen_ids.add(cid)
            unique.append(doc)

        return unique

    # ---------- 策略 7：智能上下文拼装 ----------

    def _smart_context(
        self,
        chunks: List[Document],
        scores: dict[str, float],
        max_chars: Optional[int] = None,
    ) -> Tuple[str, List[dict]]:
        """贪心按分数往缓冲区塞 chunk，总长上限 max_chars

        格式：每个 chunk 之间用 \\n\\n---\\n 分隔
        每块前缀：[来源: parent_uuid] [角标: H2标题] [相关度: 0.xx]

        Returns:
            (context_string, sources_for_citation)
        """
        max_chars = max_chars or self.max_context_chars
        pieces: List[str] = []
        sources: List[dict] = []
        used_chars = 0

        for chunk in chunks:
            text = chunk.page_content or ""
            meta = chunk.metadata

            # 构建角标前缀
            prefix_parts: List[str] = []
            pu = meta.get("parent_uuid", "")
            h2 = meta.get("Header_2", "")
            h1 = meta.get("Header_1", "")
            chunk_id = meta.get("chunk_id", "")
            score = scores.get(chunk_id, 0.0)

            if pu:
                prefix_parts.append(f"[来源: {pu[:8]}...]")
            if h1:
                prefix_parts.append(f"[H1: {h1}]")
            if h2:
                prefix_parts.append(f"[H2: {h2}]")
            if score:
                prefix_parts.append(f"[相关度: {score:.3f}]")

            prefix = " ".join(prefix_parts)
            # 总长 = 前缀 + 空行 + 正文
            piece_len = len(prefix) + 2 + len(text)

            if used_chars + piece_len > max_chars:
                continue  # 装不下就跳过

            pieces.append(f"{prefix}\n{text}")
            used_chars += piece_len

            sources.append({
                "chunk_id": chunk_id,
                "parent_uuid": pu,
                "header_1": h1,
                "header_2": h2,
                "score": score,
                "source_file": meta.get("source", ""),
                "core_topics": meta.get("core_topics", []),
                "core_scenes": meta.get("core_scenes", []),
            })

        return "\n\n---\n\n".join(pieces), sources

    # ---------- 主入口 ----------

    def retrieve(self, query: str) -> ContextBundle:
        """执行完整的三级检索管线

        Returns:
            ContextBundle（含拼装字符串、原始块、角标信息）
        """
        # 步骤 1 & 2：向量 + BM25 并行
        vec_results = self._vector_search_with_scores(query)
        bm25_results = self._bm25_search(query)

        # 步骤 3：RRF 融合
        fused = self._rrf_fuse(vec_results, bm25_results, alpha=self.alpha)

        # 步骤 4：Cosine 重排
        reranked = self._rerank_by_cosine(fused, query)

        if not reranked:
            return ContextBundle(
                query=query,
                context_text="",
                chunks=[],
                parent_map={},
                sources=[],
                reranked_scores={},
            )

        # 提取分数 map
        reranked_scores: dict[str, float] = {cid: s for cid, s, _ in reranked}
        docs = [doc for _, _, doc in reranked]

        # 步骤 5：同源去重
        unique_docs = self._dedup_and_fold(reranked)

        # 步骤 6：父子回填
        filled_docs, parent_map = self._parent_fill(unique_docs)

        # 步骤 7：智能拼装
        context_text, sources = self._smart_context(
            filled_docs, reranked_scores
        )

        return ContextBundle(
            query=query,
            context_text=context_text,
            chunks=filled_docs,
            parent_map=parent_map,
            sources=sources,
            reranked_scores=reranked_scores,
        )


# ---------- 模块级便捷函数 ----------

_default_retriever: Optional[AdvancedRetriever] = None


def advanced_retrieve(
    query: str,
    persist_directory: str = "db",
    alpha: float = 0.6,
    k: int = 8,
    max_context_chars: int = 2800,
) -> ContextBundle:
    """高级检索（RRF 融合 + cosine 重排 + 父子回填 + 智能拼装）

    Args:
        query: 用户问题
        persist_directory: 向量库路径
        alpha: RRF 中向量权重，1-alpha 为 BM25 权重
        k: 最终取 top_k 个 chunk 进入上下文
        max_context_chars: 拼装后上下文总字数上限

    Returns:
        ContextBundle 对象
    """
    global _default_retriever
    _default_retriever = AdvancedRetriever(
        persist_directory=persist_directory,
        alpha=alpha,
        k_vector=k * 2,
        k_bm25=k * 2,
        k_final=k,
        max_context_chars=max_context_chars,
    )
    return _default_retriever.retrieve(query)


# ---------- 向后兼容的旧函数 ----------

def similarity_search(
    query: str,
    k: int = 4,
    filter_dict: dict | None = None,
    persist_directory: str = "db",
) -> list[dict[str, Any]]:
    """执行相似性搜索（向后兼容）"""
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
    """格式化检索结果为可读文本（向后兼容）"""
    if not results:
        return ""

    formatted = []
    for i, result in enumerate(results, 1):
        source = result["metadata"].get("source", "未知来源")
        content = result["content"].strip()
        formatted.append(f"[文档 {i}] 来源: {source}\n{content}")

    return "\n\n---\n\n".join(formatted)


__all__ = [
    "AdvancedRetriever",
    "ContextBundle",
    "advanced_retrieve",
    "similarity_search",
    "format_search_results",
]
