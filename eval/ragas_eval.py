"""Ragas faithfulness 评估"""
import argparse
import json
import sys
import types
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# ragas 0.4.3 与新版 langchain-community 不兼容，注入占位模块避免 import 崩溃
_dummy = types.ModuleType("langchain_community.chat_models.vertexai")
_dummy.ChatVertexAI = type("ChatVertexAI", (object,), {})
sys.modules["langchain_community.chat_models.vertexai"] = _dummy
import langchain_community.llms as _lc_llms
if not hasattr(_lc_llms, "VertexAI"):
    _lc_llms.VertexAI = type("VertexAI", (object,), {})

import os
from datasets import Dataset
from langchain_deepseek import ChatDeepSeek
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import faithfulness

from eval.rag_pipeline import run_rag

DEFAULT_DATASET = Path(__file__).parent / "datasets" / "rag_eval.jsonl"
DEFAULT_OUTPUT = Path(__file__).parent / "results" / "ragas_faithfulness.json"

def load_dataset(path: Path, limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").strip().splitlines() if line.strip()]
    return rows[:limit] if limit else rows

def main():
    p = argparse.ArgumentParser(description="Ragas faithfulness 评估")
    p.add_argument("--dataset", default=str(DEFAULT_DATASET))
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p.add_argument("--retriever", choices=("baseline", "advanced"), default="baseline")
    p.add_argument("--limit", type=int, default=None, help="限制条数以控制 API 成本")
    args = p.parse_args()
    items = load_dataset(Path(args.dataset), args.limit)
    if not items:
        raise SystemExit(f"数据集为空: {args.dataset}")
    rows = []
    for i, item in enumerate(items, 1):
        print(f"[{i}/{len(items)}] RAG: {item['question'][:40]}...")
        rag = run_rag(item["question"], args.retriever)
        rows.append({"question": rag["question"], "answer": rag["answer"], "contexts": rag["contexts"]})
    ds = Dataset.from_dict({"question": [r["question"] for r in rows], "answer": [r["answer"] for r in rows], "contexts": [r["contexts"] for r in rows]})
    llm = LangchainLLMWrapper(ChatDeepSeek(model="deepseek-chat", temperature=0.1, api_key=os.getenv("DEEPSEEK_API_KEY", "")))
    result = evaluate(ds, metrics=[faithfulness], llm=llm)
    scores = result.to_pandas() if hasattr(result, "to_pandas") else result
    faith_col = "faithfulness" if hasattr(scores, "columns") and "faithfulness" in scores.columns else None
    mean_score = float(scores[faith_col].mean()) if faith_col else None
    per_item = scores[faith_col].tolist() if faith_col else []
    report = {"retriever": args.retriever, "n": len(rows), "mean_faithfulness": mean_score, "scores": per_item, "questions": [r["question"] for r in rows]}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"mean_faithfulness={mean_score:.4f}  n={len(rows)}  -> {out}")

if __name__ == "__main__":
    main()
