from __future__ import annotations

from typing import Any, Callable

from .tracer import preview

def _extract_load_inbox(state, ret, self_) -> dict:
    emails = (ret or {}).get("emails") or state.get("emails") or []
    return {"emails_fetched": len(emails)}

def _extract_categorize(state, ret, self_) -> dict:
    cat = (ret or {}).get("email_category") or state.get("email_category") or ""
    email = (ret or {}).get("current_email") or state.get("current_email")
    return {"category": cat, "email_message_id": getattr(email, "messageId", "") if email else ""}

def _extract_rag_queries(state, ret, self_) -> dict:
    queries = (ret or {}).get("rag_queries") or state.get("rag_queries") or []
    return {"query_count": len(queries), "queries": queries}

def _extract_retrieve(state, ret, self_) -> dict:
    queries = state.get("rag_queries") or []
    docs = (ret or {}).get("retrieved_documents") or ""
    return {"query_count": len(queries), "retrieved_chars": len(docs)}

def _extract_assemble(state, ret, self_) -> dict:
    ctx = (ret or {}).get("assembled_context") or state.get("assembled_context") or ""
    return {"context_chars": len(ctx), "used_sender_memory": "LONG TERM MEMORY" in ctx}

def _extract_writer(state, ret, self_) -> dict:
    trial = (ret or {}).get("trials") or state.get("trials") or 0
    email = (ret or {}).get("generated_email") or ""
    return {"trial": trial, "draft_preview": preview(email, 150)}

def _extract_proofreader(state, ret, self_) -> dict:
    sendable = (ret or {}).get("sendable")
    if sendable is None: sendable = state.get("sendable", False)
    msgs = state.get("writer_messages") or []
    feedback = msgs[-1] if msgs else ""
    if feedback.startswith("**Proofreader Feedback:**"):
        feedback = feedback.split("\n", 1)[-1]
    return {"sendable": sendable, "feedback_preview": preview(feedback, 200)}

def _extract_save_draft(state, ret, self_) -> dict:
    return {}

def _extract_skip(state, ret, self_) -> dict:
    email = state.get("current_email")
    return {
        "skipped_message_id": getattr(email, "messageId", "") if email else "",
        "subject_preview": preview(getattr(email, "subject", "") if email else ""),
    }

NODE_EXTRACTORS: dict[str, Callable[[Any, Any, Any], dict]] = {
    "load_inbox_emails": _extract_load_inbox,
    "categorize_email": _extract_categorize,
    "construct_rag_queries": _extract_rag_queries,
    "retrieve_from_rag": _extract_retrieve,
    "assemble_context": _extract_assemble,
    "email_writer": _extract_writer,
    "email_proofreader": _extract_proofreader,
    "save_draft_email": _extract_save_draft,
    "skip_unrelated_email": _extract_skip,
}
