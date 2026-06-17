"""文档加载和分割模块

采用面向对象的 DataPreparationModule：
- self.documents       : 父文档池（每个源文件一份完整内容）
- self.chunks          : 子文档池（自适应切分后的检索单元）
- self.parent_child_map: child_id → parent_uuid 映射

切分算法为自适应 Markdown 结构感知切分：
1. 按 H1/H2 粗切分
2. 超长块降级到 H3 切分
3. 极短的 H3 块通过缓冲池合并（保证上下文丰富度）
4. 超长 H3 块兜底使用字符切分
5. parent_uuid 通过文件相对路径 MD5 生成（稳定，便于按文件清理）
"""
import hashlib
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)

PROJECT_ROOT = Path(__file__).parent.parent.parent

# 分块参数
MAX_PARENT_LENGTH = 1200
MIN_PARENT_LENGTH = 400
CHUNK_OVERLAP = 30


class DataPreparationModule:
    """数据准备模块：负责文档加载 + 自适应切分 + 父子映射"""

    def __init__(
        self,
        data_path: Optional[str] = None,
        max_parent_length: int = MAX_PARENT_LENGTH,
        min_parent_length: int = MIN_PARENT_LENGTH,
    ):
        # 默认读清洗后的目录（context/clean_context/），由 data_cleaner.py 生成
        self.data_path = Path(data_path) if data_path else PROJECT_ROOT / "context" / "clean_context"
        self.max_parent_length = max_parent_length
        self.min_parent_length = min_parent_length

        self.documents: List[Document] = []             # 父文档（完整）
        self.chunks: List[Document] = []                # 子文档（检索单元）
        self.parent_child_map: Dict[str, str] = {}      # child_id → parent_uuid

        # 三把"刀"，按粒度从粗到细
        self._h2_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "Header_1"), ("##", "Header_2")],
            strip_headers=False,
        )
        self._h3_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("###", "Header_3")],
            strip_headers=False,
        )
        self._fallback_char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_parent_length,
            chunk_overlap=CHUNK_OVERLAP,
        )

    # ---------- 加载阶段 ----------

    def _make_parent_uuid(self, file_path: Path) -> str:
        """基于文件相对路径生成稳定的 parent_uuid"""
        try:
            relative_path = file_path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
        except Exception:
            relative_path = str(file_path)
        return hashlib.md5(relative_path.encode("utf-8")).hexdigest()

    def load_document(self) -> List[Document]:
        """扫描 data_path 下的 .md / .txt 文件，构造父文档列表

        每个父文档的 metadata 中包含：
        - source: 绝对文件路径
        - file_path: 同 source（兼容字段）
        - description: 文件名（不含扩展名）
        - parent_uuid: 由相对路径 MD5 生成的稳定 ID
        - doc_type: "parent"
        """
        documents: List[Document] = []
        extensions = [".md", ".txt", ".json"]

        for md_file in self.data_path.rglob("*"):
            if not md_file.is_file() or md_file.suffix not in extensions:
                continue
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                print(f"读取文件 {md_file} 失败: {e}")
                continue

            parent_uuid = self._make_parent_uuid(md_file)
            doc = Document(
                page_content=content,
                metadata={
                    "source": str(md_file),
                    "file_path": str(md_file),
                    "description": md_file.stem,
                    "parent_uuid": parent_uuid,
                    "doc_type": "parent",
                },
            )
            documents.append(doc)

        if not documents:
            print(f"警告: 未找到任何文档，{self.data_path} 目录为空或没有支持的文档格式")
            return []

        self.documents = documents
        print(f"共加载 {len(documents)} 个文档（父文档池）")
        return documents

    # ---------- 切分阶段 ----------

    def adaptive_markdown_splitter(self, document: Document) -> List[Document]:
        """自适应 Markdown 切分器（实例方法版本，依赖 self 上的配置）

        处理流程：
        1. 按 H1/H2 粗切分
        2. 长度合格（≤ max_parent_length）→ 直接入结果
        3. 超长 → 降级到 H3 切分
        4. 极短的 H3 块通过缓冲池合并
        5. 超长 H3 块兜底使用字符切分
        """
        max_len = self.max_parent_length
        min_len = self.min_parent_length

        final_parent_chunks: List[Document] = []
        md_text = document.page_content
        original_metadata = dict(document.metadata)

        h2_chunks = self._h2_splitter.split_text(md_text)

        for chunk in h2_chunks:
            chunk_length = len(chunk.page_content)
            current_h2_title = chunk.metadata.get("Header_2", "未命名二级标题")

            # 路线 A：长度合格，直接入库
            if chunk_length <= max_len:
                final_parent_chunks.append(chunk)
                continue

            # 路线 B：超长，降级到 H3
            print(f"[动态降级] 二级标题 [{current_h2_title}] 长度达 {chunk_length}，正按三级标题拆分...")
            h3_chunks = self._h3_splitter.split_text(chunk.page_content)

            # 碎片合并机制
            merged_h3_chunks: List[Document] = []
            buffer_doc: Optional[Document] = None

            for h3_chunk in h3_chunks:
                h3_chunk.metadata.update(chunk.metadata)

                # 超长 H3 块：清空缓冲池后兜底字符切分
                if len(h3_chunk.page_content) > max_len:
                    if buffer_doc is not None:
                        merged_h3_chunks.append(buffer_doc)
                        buffer_doc = None
                    print("  -> [兜底切分] 三级标题依然超长，强制字符切分。")
                    char_chunks = self._fallback_char_splitter.split_documents([h3_chunk])
                    merged_h3_chunks.extend(char_chunks)
                    continue

                # 碎片合并
                if buffer_doc is None:
                    buffer_doc = h3_chunk
                elif (len(buffer_doc.page_content) < min_len) and (
                    len(buffer_doc.page_content) + len(h3_chunk.page_content) <= max_len
                ):
                    buffer_doc.page_content += "\n\n" + h3_chunk.page_content
                else:
                    merged_h3_chunks.append(buffer_doc)
                    buffer_doc = h3_chunk

            # 收尾：缓冲池中残留的最后一个块
            if buffer_doc is not None:
                merged_h3_chunks.append(buffer_doc)

            final_parent_chunks.extend(merged_h3_chunks)

        # 为每个 chunk 写入元数据 + 建立父子映射
        parent_uuid = original_metadata.get("parent_uuid", str(uuid.uuid4()))
        for i, chunk in enumerate(final_parent_chunks):
            chunk.metadata.update(original_metadata)
            chunk_id = str(uuid.uuid4())
            chunk.metadata.update({
                "chunk_id": chunk_id,
                "parent_uuid": parent_uuid,
                "doc_type": "child",
                "chunk_index": i,
            })
            self.parent_child_map[chunk_id] = parent_uuid

        return final_parent_chunks

    def chunk_documents(self) -> List[Document]:
        """核心切分调度器：遍历 self.documents，调用 adaptive_markdown_splitter"""
        if not self.documents:
            raise ValueError("请先调用 load_document() 加载文档")

        print("🔪 正在进行智能 Markdown 结构感知分块...")
        all_chunks: List[Document] = []
        for doc in self.documents:
            doc_chunks = self.adaptive_markdown_splitter(doc)
            all_chunks.extend(doc_chunks)

        # 兜底：补齐缺失的 chunk_id / batch_index / chunk_size
        for i, chunk in enumerate(all_chunks):
            if "chunk_id" not in chunk.metadata:
                chunk.metadata["chunk_id"] = str(uuid.uuid4())
            chunk.metadata["batch_index"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)

        self.chunks = all_chunks
        print(f"✅ 分块完成：{len(self.documents)} 个父文档 → {len(all_chunks)} 个子块")
        return all_chunks

    # ---------- 兼容函数式接口（供 preprocess.py 调用） ----------

    def run_pipeline(self) -> List[Document]:
        """一键执行：load → chunk，返回最终的 chunks 列表"""
        self.load_document()
        return self.chunk_documents()


# ---------- 模块级便捷函数（保持与 preprocess.py 的兼容） ----------

_module = DataPreparationModule()


def get_documents() -> List[Document]:
    """加载所有知识文档

    注意：每次调用会重置内部状态，避免父子映射错乱。
    """
    global _module
    _module = DataPreparationModule()
    return _module.load_document()


def split_documents(documents: List[Document]) -> list[Any]:
    """对父文档列表进行自适应切分

    复用模块级 _module 的切分器，确保父子映射稳定。
    """
    if not documents:
        return []

    # 把传入的 documents 同步进模块（保留父文档池）
    _module.documents = documents
    return _module.chunk_documents()


__all__ = [
    "DataPreparationModule",
    "get_documents",
    "split_documents",
]
