"""
FaceDatabase - FAISS + SQLite storage.
Handles user management, embedding storage, and similarity search.
"""

import os
import numpy as np
import faiss
import sqlite3
from collections import defaultdict
from config import DB_DIR, TOP_K


class FaceDatabase:
    """Combined FAISS vector index + SQLite metadata store."""

    def __init__(self, db_dir=DB_DIR):
        os.makedirs(db_dir, exist_ok=True)
        self._sql_path = os.path.join(db_dir, "metadata.sqlite")
        self._idx_path = os.path.join(db_dir, "faiss.index")
        self._dim = 512

        # SQLite
        self._conn = sqlite3.connect(self._sql_path)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id    INTEGER PRIMARY KEY AUTOINCREMENT,
                name  TEXT NOT NULL,
                score REAL DEFAULT 0,
                ts    TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.commit()

        # FAISS
        if os.path.exists(self._idx_path):
            self._index = faiss.read_index(self._idx_path)
        else:
            base = faiss.IndexFlatIP(self._dim)  # cosine sim for normalized vectors
            self._index = faiss.IndexIDMap2(base)

    def _save(self):
        faiss.write_index(self._index, self._idx_path)
        self._conn.commit()

    def add_user(self, name, embeddings, scores=None):
        """Add user with embeddings. Replaces existing user."""
        self.remove_user(name)
        for i, emb in enumerate(embeddings):
            s = scores[i] if scores else 0
            cur = self._conn.execute(
                "INSERT INTO embeddings (name, score) VALUES (?, ?)", (name, s)
            )
            vec = np.array([emb], dtype=np.float32)
            self._index.add_with_ids(vec, np.array([cur.lastrowid], dtype=np.int64))
        self._save()

    def remove_user(self, name):
        """Remove all embeddings for a user."""
        rows = self._conn.execute("SELECT id FROM embeddings WHERE name=?", (name,)).fetchall()
        if rows:
            ids = np.array([r[0] for r in rows], dtype=np.int64)
            self._index.remove_ids(ids)
            self._conn.execute("DELETE FROM embeddings WHERE name=?", (name,))
            self._save()

    def match(self, query_emb):
        """Find best matching user via FAISS nearest neighbor.
        
        Returns (name, score) where score is top-K average cosine similarity.
        """
        if self._index.ntotal == 0:
            return "Unknown", 0.0

        k = min(TOP_K * 5, self._index.ntotal)
        scores, ids = self._index.search(
            np.array([query_emb], dtype=np.float32), k
        )

        # Group scores by user
        user_scores = defaultdict(list)
        for score, eid in zip(scores[0], ids[0]):
            if eid == -1:
                continue
            row = self._conn.execute(
                "SELECT name FROM embeddings WHERE id=?", (int(eid),)
            ).fetchone()
            if row:
                user_scores[row[0]].append(float(score))

        # Find user with highest top-K average
        best_name, best_score = "Unknown", 0.0
        for name, sc_list in user_scores.items():
            avg = np.mean(sorted(sc_list, reverse=True)[:TOP_K])
            if avg > best_score:
                best_score = avg
                best_name = name

        return best_name, best_score

    def get_users(self):
        """Get dict {name: embedding_count}."""
        rows = self._conn.execute(
            "SELECT name, COUNT(*) FROM embeddings GROUP BY name"
        ).fetchall()
        return {n: c for n, c in rows}

    @property
    def total(self):
        return self._index.ntotal

    def close(self):
        self._conn.close()
