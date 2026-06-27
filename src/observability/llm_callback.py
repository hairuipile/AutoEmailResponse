from __future__ import annotations

from typing import Any
from uuid import uuid4

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from pydantic import BaseModel

from .tracer import current_span_id, current_trace_id, write_record

class TraceCallbackHandler(BaseCallbackHandler):
    def on_llm_end(self, response: LLMResult, *, run_id, parent_run_id=None, **kwargs: Any) -> None:
        tid = current_trace_id()
        if not tid: return
        usage = (response.llm_output or {}).get("token_usage") or {}
        if not usage and response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    info = getattr(gen, "generation_info", None) or {}
                    usage = info.get("token_usage") or info
                    if usage: break
                if usage: break
        model = (response.llm_output or {}).get("model_name") or (response.llm_output or {}).get("model") or ""
        write_record({
            "type": "span", "trace_id": tid, "span_id": str(uuid4()),
            "parent_span_id": current_span_id(), "name": "llm",
            "input_tokens": usage.get("prompt_tokens") or usage.get("input_tokens") or 0,
            "output_tokens": usage.get("completion_tokens") or usage.get("output_tokens") or 0,
            "model": model, "status": "ok",
        })

    def on_chain_end(self, outputs: dict, *, run_id, parent_run_id=None, **kwargs: Any) -> None:
        tid = current_trace_id()
        if not tid or not outputs: return
        extra = {}
        for v in outputs.values() if isinstance(outputs, dict) else [outputs]:
            if not isinstance(v, BaseModel):
                continue
            if hasattr(v, "category"):
                extra["structured_category"] = getattr(v.category, "value", str(v.category))
            if hasattr(v, "send"):
                extra["structured_send"] = v.send
            if hasattr(v, "queries"):
                extra["structured_queries"] = v.queries
        if extra:
            write_record({"type": "event", "trace_id": tid, "name": "structured_output", **extra})
