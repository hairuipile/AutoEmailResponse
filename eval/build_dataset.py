"""从 Chroma 抽样 chunk，LLM 合成 QA 评估集"""
import argparse
import json
import random
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).parent.parent / ".env")

import os
import chromadb

DEFAULT_OUTPUT = Path(__file__).parent / "datasets" / "rag_eval.jsonl"

class SynthQA(BaseModel):
    question: str = Field(..., description="基于 chunk 的可检索问题")
    reference_answer: str = Field(..., description="基于 chunk 的简短参考答案")

SYNTH_PROMPT = ChatPromptTemplate.from_template(
    "根据以下知识片段，生成 1 个用户可能提出的检索问题，以及仅基于该片段的简短参考答案（不要引入片段外信息）。\n\n片段：\n{chunk}"
)

def sample_chunks(limit: int) -> list[dict]:
    db_path = Path(__file__).parent.parent / "db"
    if not db_path.exists():
        raise SystemExit("向量库不存在，请先运行 python main.py 构建 db/")
    col = chromadb.PersistentClient(path=str(db_path)).get_or_create_collection("langchain")
    rows = col.get(include=["documents", "metadatas"])
    docs, metas = rows.get("documents") or [], rows.get("metadatas") or []
    chunks = [{"text": d, "meta": m} for d, m in zip(docs, metas) if m and m.get("doc_type") == "child" and m.get("chunk_id")]
    if not chunks:
        chunks = [{"text": d, "meta": m} for d, m in zip(docs, metas) if m and m.get("chunk_id")]
    if not chunks:
        raise SystemExit("向量库无 chunk，请先构建索引")
    random.shuffle(chunks)
    return chunks[:limit]

def main():
    p = argparse.ArgumentParser(description="合成 RAG 评估数据集")
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    random.seed(args.seed)
    llm = ChatDeepSeek(model="deepseek-chat", temperature=0.3, api_key=os.getenv("DEEPSEEK_API_KEY", ""))
    chain = SYNTH_PROMPT | llm.with_structured_output(SynthQA)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, item in enumerate(sample_chunks(args.limit), 1):
        text, meta = item["text"], item["meta"]
        chunk_id = meta["chunk_id"]
        try:
            qa = chain.invoke({"chunk": text[:2000]})
            row = {
                "question": qa.question,
                "relevant_chunk_ids": [chunk_id],
                "reference_answer": qa.reference_answer,
                "source_file": meta.get("source") or meta.get("file_path") or "",
            }
            lines.append(json.dumps(row, ensure_ascii=False))
            print(f"[{i}/{args.limit}] {qa.question[:50]}...")
        except Exception as e:
            print(f"[{i}/{args.limit}] 跳过: {e}")
    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"已写入 {len(lines)} 条 -> {out}")

if __name__ == "__main__":
    main()
