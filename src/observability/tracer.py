from __future__ import annotations



import json

import os

import time

import uuid

from contextvars import ContextVar

from functools import wraps

from pathlib import Path

from typing import Any



_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)

_span_stack: ContextVar[list[str]] = ContextVar("span_stack", default=[])

_span_annotations: ContextVar[list[dict]] = ContextVar("span_annotations", default=[])

_trace_start_ms: ContextVar[float | None] = ContextVar("trace_start_ms", default=None)

_trace_starts: dict[str, float] = {}



def _enabled() -> bool:

    return os.getenv("TRACE_ENABLED", "true").lower() in ("1", "true", "yes")



def _log_path() -> Path:

    return Path(os.getenv("TRACE_LOG_PATH", "logs/traces.jsonl"))



def preview(text: str | None, n: int = 200) -> str:

    if not text: return ""

    t = str(text).replace("\n", " ").strip()

    return t if len(t) <= n else t[:n] + "..."



def write_record(record: dict[str, Any]) -> None:

    if not _enabled(): return

    path = _log_path()

    path.parent.mkdir(parents=True, exist_ok=True)

    record.setdefault("ts_ms", int(time.time() * 1000))

    with path.open("a", encoding="utf-8") as f:

        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")



def resolve_trace_id(state=None, explicit: str | None = None) -> str | None:

    tid = explicit or (state.get("trace_id") if state else None) or _trace_id.get()

    return tid or None



def bind_trace(trace_id: str | None) -> None:

    if trace_id:

        _trace_id.set(trace_id)



def start_email_trace(email: Any) -> str:

    if not _enabled(): return ""

    tid = str(uuid.uuid4())

    _trace_id.set(tid)

    _span_stack.set([])

    t0 = time.perf_counter()

    _trace_start_ms.set(t0)

    _trace_starts[tid] = t0

    write_record({

        "type": "trace_start",

        "trace_id": tid,

        "email_message_id": getattr(email, "messageId", "") or "",

        "email_sender": getattr(email, "sender", "") or "",

        "email_subject": getattr(email, "subject", "") or "",

        "email_body_preview": preview(getattr(email, "body", "")),

    })

    return tid



def end_email_trace(status: str = "success", metadata: dict | None = None, trace_id: str | None = None, state=None) -> None:

    if not _enabled(): return

    tid = resolve_trace_id(state, trace_id)

    if not tid: return

    start = _trace_starts.pop(tid, None) or _trace_start_ms.get()

    duration_ms = round((time.perf_counter() - start) * 1000, 2) if start else None

    rec = {"type": "trace_end", "trace_id": tid, "status": status, "duration_ms": duration_ms}

    if metadata: rec.update(metadata)

    write_record(rec)

    if _trace_id.get() == tid:

        _trace_id.set(None)

        _span_stack.set([])

        _trace_start_ms.set(None)



def record_event(name: str, data: dict | None = None, trace_id: str | None = None, state=None) -> None:

    if not _enabled(): return

    rec = {"type": "event", "trace_id": resolve_trace_id(state, trace_id), "name": name}

    if data: rec.update(data)

    write_record(rec)



def annotate(data: dict) -> None:

    if not _enabled(): return

    anns = _span_annotations.get()

    anns.append(data)

    _span_annotations.set(anns)



def traced_node(name: str):

    def deco(fn):

        @wraps(fn)

        def wrapper(self, state, *args, **kwargs):

            if not _enabled():

                return fn(self, state, *args, **kwargs)

            from .extractors import NODE_EXTRACTORS

            bind_trace(state.get("trace_id") if isinstance(state, dict) else None)

            span_id = str(uuid.uuid4())

            stack = _span_stack.get()

            parent = stack[-1] if stack else None

            _span_stack.set(stack + [span_id])

            ann_token = _span_annotations.set([])

            t0 = time.perf_counter()

            status, err, ret = "ok", None, None

            try:

                ret = fn(self, state, *args, **kwargs)

                return ret

            except Exception as e:

                status, err = "error", str(e)

                raise

            finally:

                extra = {}

                if name in NODE_EXTRACTORS:

                    try:

                        extra = NODE_EXTRACTORS[name](state, ret, self) or {}

                    except Exception:

                        pass

                anns = list(_span_annotations.get([]))

                rec = {

                    "type": "span", "trace_id": resolve_trace_id(state), "span_id": span_id,

                    "parent_span_id": parent, "name": name,

                    "duration_ms": round((time.perf_counter() - t0) * 1000, 2),

                    "status": status, **extra,

                }

                if anns: rec["annotations"] = anns

                if err: rec["error"] = err

                write_record(rec)

                _span_stack.set(stack)

                _span_annotations.reset(ann_token)

        return wrapper

    return deco



def current_trace_id() -> str | None:

    return _trace_id.get()



def current_span_id() -> str | None:

    stack = _span_stack.get()

    return stack[-1] if stack else None

