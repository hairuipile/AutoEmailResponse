from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..state import GraphState
from ..memory.sender_memory import SenderMemoryManager, SenderEpisode


class ContextManager:
    def __init__(
        self,
        sender_memory: SenderMemoryManager | None = None,
        max_token_limit: int = 400,
        rules_path: str | Path = "context/company_rules.md",
        long_term_token_budget: int = 800,
        short_term_token_budget: int = 600,
        rag_token_budget: int = 600,
        relevant_episode_top_k: int = 5,
        relevant_episode_candidate_limit: int = 20,
    ) -> None:
        self.sender_memory = sender_memory
        self.max_token_limit = max_token_limit
        self.rules_path = Path(rules_path)
        self.long_term_token_budget = long_term_token_budget
        self.short_term_token_budget = short_term_token_budget
        self.rag_token_budget = rag_token_budget
        self.relevant_episode_top_k = relevant_episode_top_k
        self.relevant_episode_candidate_limit = relevant_episode_candidate_limit

    def build_context_bundle(
        self,
        state: GraphState,
        sender_memory: SenderMemoryManager | None = None,
    ) -> str:
        top_level_rules = self._load_top_level_rules()
        long_term_context = self._build_long_term_context(state, sender_memory)
        rag_context = self._build_rag_context(state)
        short_term_context = self._build_short_term_context(state)
        selected_context = (state.get("selected_context") or "").strip()
        return self._build_bundle(
            top_level_rules=top_level_rules,
            long_term_context=long_term_context,
            rag_context=rag_context,
            short_term_context=short_term_context,
            selected_context=selected_context,
        )

    def _load_top_level_rules(self) -> str:
        if not self.rules_path.exists():
            return ""
        text = self.rules_path.read_text(encoding="utf-8")
        return text.strip()

    def _build_long_term_context(
        self,
        state: GraphState,
        sender_memory: SenderMemoryManager | None,
    ) -> str:
        sender_key = state.get("sender_key") or (
            state["current_email"].sender if "current_email" in state else ""
        )
        if not sender_key or sender_memory is None:
            return ""

        strategy = sender_memory.build_strategy(sender_key)
        query = self._build_memory_query(state["current_email"])
        episodes = sender_memory.query_relevant_episodes(
            sender_key,
            query,
            top_k=self.relevant_episode_top_k,
            candidate_limit=self.relevant_episode_candidate_limit,
        )
        episode_text = self._format_episodes(episodes)
        if not strategy and not episode_text:
            return ""
        pieces: list[str] = []
        if strategy:
            pieces.append(strategy)
        if episode_text:
            pieces.append(episode_text)
        context = "\n\n".join(pieces)
        return self._fit_to_budget(context, self.long_term_token_budget)

    def _build_rag_context(self, state: GraphState) -> str:
        retrieved_documents = (state.get("retrieved_documents") or "").strip()
        if not retrieved_documents:
            return ""
        return self._fit_to_budget(retrieved_documents, self.rag_token_budget)

    def _build_short_term_context(self, state: GraphState) -> str:
        selected_context = (state.get("selected_context") or "").strip()
        context_summary = (state.get("context_summary") or "").strip()
        writer_messages = state.get("writer_messages") or []
        writer_history = self._messages_to_text(writer_messages)
        pieces = [part for part in [writer_history, context_summary, selected_context] if part]
        summary = "\n\n".join(pieces)
        return self._fit_to_budget(summary, self.short_term_token_budget)

    def _build_bundle(
        self,
        *,
        top_level_rules: str | None = None,
        long_term_context: str | None = None,
        rag_context: str | None = None,
        short_term_context: str | None = None,
        selected_context: str | None = None,
    ) -> str:
        sections: list[str] = []
        if top_level_rules:
            sections.append(self._section("TOP LEVEL RULES", top_level_rules))
        if long_term_context:
            sections.append(self._section("LONG TERM MEMORY", long_term_context))
        if rag_context:
            sections.append(self._section("RAG KNOWLEDGE", rag_context))
        if short_term_context:
            sections.append(self._section("SHORT TERM MEMORY", short_term_context))
        if selected_context:
            sections.append(self._section("SELECTED CONTEXT", selected_context))
        if not sections:
            return ""
        bundle = "\n\n".join(sections)
        return self._fit_to_budget(bundle, self.max_token_limit)

    def _build_memory_query(self, current_email: Any) -> str:
        subject = getattr(current_email, "subject", "")
        body = getattr(current_email, "body", "")
        parts = [part for part in [subject, body] if part]
        return "\n".join(parts)

    def _messages_to_text(self, messages: list[Any]) -> str:
        parts = []
        for message in messages:
            content = getattr(message, "content", message)
            parts.append(str(content).strip())
        return "\n\n".join(part for part in parts if part)

    def _format_episodes(self, episodes: list[SenderEpisode]) -> str:
        if not episodes:
            return ""
        formatted = []
        for episode in episodes:
            metadata_summary = ", ".join(f"{key}={value}" for key, value in episode.metadata.items())
            line = f"- [{episode.occurred_at}] category={episode.category}, sendable={episode.sendable}, summary={episode.summary}"
            if metadata_summary:
                line = f"{line}, metadata={metadata_summary}"
            formatted.append(line)
        return "\n".join(formatted)

    def _section(self, title: str, content: str) -> str:
        cleaned = content.strip()
        return f"# {title}\n\n{cleaned}"

    def _fit_to_budget(self, text: str, budget: int) -> str:
        text = text.strip()
        if not text:
            return ""
        tokens = self._estimate_tokens(text)
        if tokens <= budget:
            return text
        keep_ratio = budget / tokens
        return self._trim_text(text, keep_ratio)

    def _trim_text(self, text: str, keep_ratio: float) -> str:
        keep_ratio = max(0.2, min(1.0, keep_ratio))
        target_length = max(1, int(len(text) * keep_ratio))
        trimmed = text[:target_length].rstrip()
        return f"{trimmed}\n\n[Context truncated for token budget]"

    def _estimate_tokens(self, text: str) -> int:
        tokens = re.findall(r"\b\w+\b", text)
        return len(tokens)
