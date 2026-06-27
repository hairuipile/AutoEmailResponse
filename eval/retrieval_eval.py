"""检索评估：Recall@K + MRR，对比 baseline vs advanced"""
import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from eval.rag_pipeline import retrieve_chunk_ids

K = 3
DEFAULT_DATASET = Path(__file__).parent / "datasets" / "rag_eval.jsonl"
DEFAULT_OUTPUT = Path(__file__).parent / "results" / "retrieval_report.json"

def load_dataset(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").strip().splitlines() if line.strip()]

def recall_at_k(retrieved: list[str], gold: set[str], k: int) -> bool:
    return bool(gold & set(retrieved[:k]))

def reciprocal_rank(retrieved: list[str], gold: set[str]) -> float:
    for i, cid in enumerate(retrieved, 1):
        if cid in gold: return 1.0 / i
    return 0.0

def eval_mode(items: list[dict], mode: str, k: int) -> dict:
    recalls, rrs = [], []
    for item in items:
        gold = set(item.get("relevant_chunk_ids") or [])
        if not gold: continue
        ids = retrieve_chunk_ids(item["question"], mode, k)
        recalls.append(recall_at_k(ids, gold, k))
        rrs.append(reciprocal_rank(ids, gold))
    n = len(recalls)
    return {"mode": mode, "k": k, "n": n, "recall_at_k": round(sum(recalls) / n, 4) if n else 0, "mrr": round(sum(rrs) / n, 4) if n else 0}

def main():
    p = argparse.ArgumentParser(description="RAG 检索评估 Recall@K + MRR")
    p.add_argument("--dataset", default=str(DEFAULT_DATASET))
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p.add_argument("--k", type=int, default=K)
    args = p.parse_args()
    items = load_dataset(Path(args.dataset))
    if not items:
        raise SystemExit(f"数据集为空: {args.dataset}")
    report = {"dataset": args.dataset, "results": [eval_mode(items, m, args.k) for m in ("baseline", "advanced")]}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for r in report["results"]:
        print(f"{r['mode']:10} Recall@{r['k']}={r['recall_at_k']:.4f}  MRR={r['mrr']:.4f}  n={r['n']}")
    print(f"报告已写入 {out}")

if __name__ == "__main__":
    main()
