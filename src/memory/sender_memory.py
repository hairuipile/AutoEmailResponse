from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Protocol, runtime_checkable


@dataclass(frozen=True)
class SenderEpisode:
    sender: str
    category: str
    sendable: bool
    summary: str
    occurred_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(
        cls,
        sender: str,
        category: str,
        sendable: bool,
        summary: str,
        occurred_at: str,
        metadata: Dict[str, Any] | None = None,
    ) -> "SenderEpisode":
        return cls(
            sender=sender,
            category=category,
            sendable=sendable,
            summary=summary,
            occurred_at=occurred_at,
            metadata=metadata or {},
        )


@runtime_checkable
class MemoryStore(Protocol):
    def add_episode(self, sender: str, episode: SenderEpisode) -> None:
        ...

    def list_episodes(self, sender: str) -> List[SenderEpisode]:
        ...


class DatabaseMemoryStore:
    def __init__(self, database_path: str | Path = "db/sender_memory.db") -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _initialize(self) -> None:
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sender_episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sender TEXT NOT NULL,
                    category TEXT NOT NULL,
                    sendable INTEGER NOT NULL,
                    summary TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    metadata TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_sender_episodes_sender ON sender_episodes(sender)"
            )

    def add_episode(self, sender: str, episode: SenderEpisode) -> None:
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                INSERT INTO sender_episodes (sender, category, sendable, summary, occurred_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    sender,
                    episode.category,
                    1 if episode.sendable else 0,
                    episode.summary,
                    episode.occurred_at,
                    json.dumps(episode.metadata, ensure_ascii=False),
                ),
            )

    def list_episodes(self, sender: str) -> List[SenderEpisode]:
        with sqlite3.connect(self.database_path) as connection:
            cursor = connection.execute(
                """
                SELECT category, sendable, summary, occurred_at, metadata
                FROM sender_episodes
                WHERE sender = ?
                ORDER BY occurred_at ASC
                """,
                (sender,),
            )
            episodes = []
            for category, sendable, summary, occurred_at, metadata in cursor.fetchall():
                episodes.append(
                    SenderEpisode(
                        sender=sender,
                        category=category,
                        sendable=bool(sendable),
                        summary=summary,
                        occurred_at=occurred_at,
                        metadata=json.loads(metadata),
                    )
                )
            return episodes


class SenderMemoryManager:
    def __init__(self, store: MemoryStore | None = None):
        self.store = store

    def load_history(self, sender_key: str) -> List[SenderEpisode]:
        if self.store is None:
            return []
        return self.store.list_episodes(sender_key)

    def build_strategy(self, sender_key: str) -> str:
        episodes = self.load_history(sender_key)
        if not episodes:
            return (
                "No prior memory for this sender. Use neutral professional tone, confirm intent first, "
                "and prefer concise replies."
            )
        latest = episodes[-1]
        metadata_hints = self._format_metadata_hints(latest.metadata)
        if latest.sendable:
            return (
                f"This sender recently received a successful reply for {latest.category}. "
                f"Maintain consistent tone and keep continuity with prior context."
                f"{metadata_hints}"
            )
        return (
            f"This sender's last interaction was not successfully sent. "
            f"Double-check tone and completeness before replying again."
            f"{metadata_hints}"
        )

    def build_long_term_memory(self, sender_key: str, max_episodes: int = 5) -> str:
        episodes = self.load_history(sender_key)[-max_episodes:]
        if not episodes:
            return ""
        lines = []
        for episode in episodes:
            metadata_summary = self._format_metadata_summary(episode.metadata)
            line = f"- [{episode.occurred_at}] category={episode.category}, summary={episode.summary}"
            if metadata_summary:
                line = f"{line}, metadata={metadata_summary}"
            lines.append(line)
        return "\n".join(lines)

    def query_relevant_episodes(
        self,
        sender_key: str,
        query: str,
        top_k: int = 5,
        candidate_limit: int = 20,
    ) -> List[SenderEpisode]:
        episodes = self.load_history(sender_key)
        if not episodes or not query.strip():
            return []

        query_tokens = self._tokenize(query)
        query_terms = set(query_tokens)
        if not query_terms:
            return []

        scored = []
        for episode in episodes:
            lexical_score = self._score_lexical(episode, query_terms)
            metadata_score = self._score_metadata(episode, query_terms)
            status_score = 1.0 if episode.sendable else -0.5
            recency_score = self._score_recency(episode.occurred_at)
            total_score = lexical_score + metadata_score + status_score + recency_score
            scored.append((total_score, episode))

        scored.sort(key=lambda item: item[0], reverse=True)
        candidates = [episode for _, episode in scored[:candidate_limit]]
        ranked = self._rerank(candidates, query)
        return ranked[:top_k]

    def save_episode(self, sender_key: str, episode: SenderEpisode) -> None:
        if self.store is not None:
            self.store.add_episode(sender_key, episode)

    def _format_metadata_summary(self, metadata: Dict[str, Any]) -> str:
        if not metadata:
            return ""
        parts = [f"{key}={value}" for key, value in metadata.items()]
        return ", ".join(parts)

    def _format_metadata_hints(self, metadata: Dict[str, Any]) -> str:
        if not metadata:
            return ""
        parts = []
        if "category" in metadata:
            parts.append(f"prior category was {metadata['category']}")
        if "subject" in metadata:
            parts.append(f"prior subject was {metadata['subject']}")
        if "user_id" in metadata:
            parts.append(f"this sender is associated with user {metadata['user_id']}")
        if "thread_id" in metadata:
            parts.append(f"prior conversation thread id was {metadata['thread_id']}")
        if "email_id" in metadata:
            parts.append(f"prior email id was {metadata['email_id']}")
        if "references" in metadata:
            parts.append(f"prior references were {metadata['references']}")
        if not parts:
            return ""
        return " Additional context: " + "; ".join(parts) + "."

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def _score_lexical(self, episode: SenderEpisode, query_terms: set[str]) -> float:
        if not query_terms:
            return 0.0
        episode_text = " ".join(
            filter(
                None,
                [
                    episode.summary,
                    episode.category,
                    episode.occurred_at,
                    *[str(value) for value in episode.metadata.values()],
                ],
            )
        )
        episode_tokens = set(self._tokenize(episode_text))
        overlap = len(query_terms & episode_tokens)
        return overlap / max(len(query_terms), 1)

    def _score_metadata(self, episode: SenderEpisode, query_terms: set[str]) -> float:
        if not query_terms or not episode.metadata:
            return 0.0
        metadata_text = " ".join(str(value) for value in episode.metadata.values())
        metadata_tokens = set(self._tokenize(metadata_text))
        overlap = len(query_terms & metadata_tokens)
        return overlap / max(len(query_terms), 1)

    def _score_recency(self, occurred_at: str) -> float:
        try:
            occurred = datetime.fromisoformat(occurred_at)
            now = datetime.now(timezone.utc)
            delta = now - occurred
            days = max(delta.total_seconds() / 86400.0, 0.0)
            if days <= 7:
                return 1.0
            if days <= 30:
                return 0.5
            if days <= 90:
                return 0.25
            return 0.0
        except Exception:
            return 0.0

    def _rerank(self, episodes: List[SenderEpisode], query: str) -> List[SenderEpisode]:
        if not episodes:
            return []
        query_terms = set(self._tokenize(query))

        def sort_key(episode: SenderEpisode):
            lexical = self._score_lexical(episode, query_terms)
            metadata = self._score_metadata(episode, query_terms)
            recency = self._score_recency(episode.occurred_at)
            status = 1.0 if episode.sendable else 0.0
            return (-lexical, -metadata, -recency, -status)

        return sorted(episodes, key=sort_key)
