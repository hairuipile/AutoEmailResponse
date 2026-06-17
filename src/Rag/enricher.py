"""元数据增强模块：对文档块进行摘要/标签等元数据注入"""
import hashlib
import json
import os
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSerializable
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent


class MetadataEnricher:
    """
    通用元数据增强器：
    - 基于 LLM 提取关键词、摘要、标签
    - 本地 JSON 文件缓存，命中则跳过
    """

    def __init__(self, llm: RunnableSerializable, cache_path: str = "metadata_cache.json"):
        self.llm = llm
        self.cache_path = PROJECT_ROOT / cache_path
        self.cache_db = self._load_cache()
        print(f"   -> [Enricher] 缓存加载完毕，当前记忆库已有 {len(self.cache_db)} 条记录。")

        self.summary_prompt = PromptTemplate.from_template(
            """你是一个专业的知识库内容分析专家。
【原文】
{text}

请为这段文本提取结构化元数据，输出 JSON 格式：
{{"summary": "简短摘要（不超过50字）", "keywords": ["关键词1", "关键词2", "关键词3"]}}
只输出 JSON，不要其他内容。"""
        )

        self.summary_chain = self.summary_prompt | self.llm | StrOutputParser()

    def _load_cache(self) -> dict:
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_cache(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache_db, f, ensure_ascii=False, indent=4)

    def _extract_candidates(self, text: str) -> list[str]:
        words = re.findall(r"[\u4e00-\u9fff]{2,}|[\w]{3,}", text)
        freq = {}
        for w in words:
            lw = w.lower()
            freq[lw] = freq.get(lw, 0) + 1
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in sorted_words[:20]]

    def _enrich_one(self, chunk: Document) -> dict:
        text = chunk.page_content
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()

        if text_hash in self.cache_db:
            return self.cache_db[text_hash]

        candidates = self._extract_candidates(text)
        raw_str = ", ".join(candidates) if candidates else "无关键词"

        try:
            response = self.summary_chain.invoke({"text": text[:2000], "raw_entities": raw_str})
            content = response.strip()
            content = re.sub(r"^```json\s*|```$", "", content, flags=re.IGNORECASE)
            meta = json.loads(content)
        except Exception as e:
            meta = {"summary": "", "keywords": []}

        self.cache_db[text_hash] = meta
        return meta

    def enrich(self, chunks: list[Document]) -> list[Document]:
        has_new = False
        print(f"\n🔍 [Enricher] 开始执行元数据增强流水线，共 {len(chunks)} 个文档块...")

        for chunk in tqdm(chunks, desc="🏷️  元数据增强中", unit="块", colour="green"):
            meta = self._enrich_one(chunk)
            chunk.metadata.update(meta)
            chunk.metadata["is_enriched"] = True
            chunk.metadata["from_cache"] = chunk.page_content in self.cache_db

        if has_new or any(not c.metadata.get("from_cache") for c in chunks):
            self._save_cache()
            if all(c.metadata.get("from_cache") for c in chunks):
                print("\n⚡ [Enricher] 所有区块均命中缓存，极速处理完毕。")
            else:
                print("\n💾 [Enricher] 发现新知识，已成功更新本地持久化缓存。")

        return chunks


def enrich_chunks(chunks: list[Document], llm: RunnableSerializable) -> list[Document]:
    """快捷入口：直接对 chunks 做元数据增强（用于 preprocess 流程）"""
    enricher = MetadataEnricher(llm=llm)
    return enricher.enrich(chunks)
