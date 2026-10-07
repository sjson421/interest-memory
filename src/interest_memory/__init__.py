"""Learns what a reader likes from weighted feedback and scores new text by it."""

import sqlite3
from collections.abc import Callable, Sequence
from datetime import datetime

import numpy as np

Embed = Callable[[Sequence[str]], np.ndarray]


def _fastembed() -> Embed:
    """Load the default local embedding model. Imported lazily so the model loads only on first use."""
    from fastembed import TextEmbedding

    model = TextEmbedding("BAAI/bge-small-en-v1.5")
    return lambda texts: np.array(list(model.embed(list(texts), batch_size=16)))


def _unit(vecs: np.ndarray) -> np.ndarray:
    """Scale each row to length 1. All-zero rows stay zero."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms == 0, 1, norms)


class InterestMemory:
    """Weighted feedback events stored in SQLite, used to score new text."""

    def __init__(self, path: str, k: int = 10, half_life_days: float = 60.0, embed: Embed | None = None):
        """Open or create the database at `path`. `embed` overrides the default fastembed model."""
        if k < 1 or half_life_days <= 0:
            raise ValueError("k must be >= 1 and half_life_days must be > 0")
        self.k = k
        self.half_life_s = half_life_days * 86400
        self._embed = embed
        self.db = sqlite3.connect(path)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS events (
                item_id TEXT PRIMARY KEY,
                vec BLOB NOT NULL,
                weight REAL NOT NULL,
                at REAL  -- NULL for seeds, which never decay
            );
            CREATE TABLE IF NOT EXISTS profile (
                version INTEGER PRIMARY KEY,
                text TEXT NOT NULL
            );
        """)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return one unit-length float32 vector per text."""
        if self._embed is None:
            self._embed = _fastembed()
        return _unit(np.asarray(self._embed(texts), dtype=np.float32))

    def add_seeds(self, phrases: Sequence[str], weight: float = 1.0) -> None:
        """Store interest phrases that never decay. Each is stored under the id `seed:<phrase>`."""
        if not phrases:
            return
        vecs = self.embed(phrases)
        with self.db:
            self.db.executemany(
                "INSERT OR REPLACE INTO events VALUES (?, ?, ?, NULL)",
                [(f"seed:{p}", v.tobytes(), weight) for p, v in zip(phrases, vecs)],
            )

    def record(self, item_id: str, text: str, weight: float, at: datetime) -> None:
        """One label per item: a later call for the same id replaces the earlier one."""
        vec = self.embed([text])[0]
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO events VALUES (?, ?, ?, ?)",
                (item_id, vec.tobytes(), weight, at.timestamp()),
            )

    def forget(self, item_id: str) -> None:
        """Delete the event for `item_id`, undoing `record` or `add_seeds`."""
        with self.db:
            self.db.execute("DELETE FROM events WHERE item_id = ?", (item_id,))

    def score(self, texts: Sequence[str], now: datetime) -> list[float]:
        """Sum of weight x decay x similarity over the k most similar events, divided by k."""
        if not texts:
            return []
        rows = self.db.execute("SELECT vec, weight, at FROM events").fetchall()
        if not rows:
            return [0.0] * len(texts)
        # ponytail: brute-force scan of every event, add an ANN index if events reach ~1M
        vecs = np.stack([np.frombuffer(v, dtype=np.float32) for v, _, _ in rows])
        age = np.array([0.0 if at is None else max(now.timestamp() - at, 0.0) for _, _, at in rows])
        weights = np.array([w for _, w, _ in rows]) * 0.5 ** (age / self.half_life_s)

        # Clip so a dissimilar rejected event cannot add to the score (negative x negative).
        sims = np.maximum(self.embed(texts) @ vecs.T, 0.0)
        k = min(self.k, len(rows))
        nearest = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        picked = np.take_along_axis(sims, nearest, axis=1) * weights[nearest]
        return (picked.sum(axis=1) / self.k).tolist()

    def profile(self) -> str:
        """Return the latest profile text, or "" if none is set."""
        row = self.db.execute("SELECT text FROM profile ORDER BY version DESC LIMIT 1").fetchone()
        return row[0] if row else ""

    def set_profile(self, text: str) -> None:
        """Save `text` as the newest profile version. Earlier versions are kept."""
        with self.db:
            self.db.execute("INSERT INTO profile (text) VALUES (?)", (text,))
