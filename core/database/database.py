"""
FaceDatabase v5.0 - FAISS + SQLite storage.
Enhanced: Prototype matching, dual-strategy scoring, gallery update policy.
"""

import os
import numpy as np
import faiss
import sqlite3
from collections import defaultdict
from config import DB_DIR, TOP_K, PROTOTYPE_ENABLED, PROTOTYPE_WEIGHT
from logger import get_logger

logger = get_logger("database")


class FaceDatabase:
    """Combined FAISS vector index + SQLite metadata + Prototype store."""

    def __init__(self, db_dir=DB_DIR):
        os.makedirs(db_dir, exist_ok=True)
        self._sql_path = os.path.join(db_dir, "metadata.sqlite")
        self._idx_path = os.path.join(db_dir, "faiss.index")
        self._proto_path = os.path.join(db_dir, "prototypes.npy")
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
            base = faiss.IndexFlatIP(self._dim)
            self._index = faiss.IndexIDMap2(base)

        # Prototype store: { name: numpy_512d }
        self._prototypes = {}
        self._proto_matrix = None
        self._proto_names = []
        
        if os.path.exists(self._proto_path):
            try:
                data = np.load(self._proto_path, allow_pickle=True).item()
                self._prototypes = data
                logger.info(f"Loaded {len(self._prototypes)} user prototypes")
                self._update_proto_cache()
            except Exception:
                self._prototypes = {}

    def _update_proto_cache(self):
        """Tạo ma trận numpy cache cho prototypes để dùng phép nhân ma trận."""
        if self._prototypes:
            self._proto_matrix = np.array(list(self._prototypes.values()), dtype=np.float32)
            self._proto_names = list(self._prototypes.keys())
        else:
            self._proto_matrix = None
            self._proto_names = []

    def _save(self):
        faiss.write_index(self._index, self._idx_path)
        self._conn.commit()

    def _save_prototypes(self):
        """Persist prototypes to disk."""
        np.save(self._proto_path, self._prototypes)

    def add_user(self, name, embeddings, scores=None, prototype=None):
        """Add user with embeddings + prototype. Replaces existing user."""
        self.remove_user(name)
        for i, emb in enumerate(embeddings):
            s = scores[i] if scores else 0
            cur = self._conn.execute(
                "INSERT INTO embeddings (name, score) VALUES (?, ?)", (name, s)
            )
            vec = np.array([emb], dtype=np.float32)
            self._index.add_with_ids(vec, np.array([cur.lastrowid], dtype=np.int64))
        
        # Lưu prototype cho user
        if prototype is not None:
            self._prototypes[name] = np.array(prototype, dtype=np.float32)
        elif embeddings:
            # Auto-compute prototype nếu không được cung cấp
            embs = np.array(embeddings, dtype=np.float32)
            proto = embs.mean(axis=0)
            proto = proto / np.linalg.norm(proto)
            self._prototypes[name] = proto
        
        self._save()
        self._save_prototypes()
        self._update_proto_cache()

    def remove_user(self, name):
        """Remove all embeddings + prototype for a user."""
        rows = self._conn.execute("SELECT id FROM embeddings WHERE name=?", (name,)).fetchall()
        if rows:
            ids = np.array([r[0] for r in rows], dtype=np.int64)
            self._index.remove_ids(ids)
            self._conn.execute("DELETE FROM embeddings WHERE name=?", (name,))
            self._save()
        
        # Xóa prototype
        if name in self._prototypes:
            del self._prototypes[name]
            self._save_prototypes()
            self._update_proto_cache()

    def match(self, query_emb):
        """Find best matching user via dual-strategy matching.
        Delegates to shared MatchingEngine (BUG-09 fix).
        """
        from models.matching import MatchingEngine
        from config import (PROTOTYPE_ENABLED, PROTOTYPE_WEIGHT, 
                          PROTOTYPE_MODE, PROTOTYPE_REJECT_THRESHOLD,
                          THRESHOLD_REJECT, COHORT_ENABLED, COHORT_Z_THRESHOLD)
        engine = MatchingEngine(
            top_k=TOP_K,
            proto_weight=PROTOTYPE_WEIGHT,
            proto_enabled=PROTOTYPE_ENABLED,
            proto_mode=PROTOTYPE_MODE,
            proto_reject_threshold=PROTOTYPE_REJECT_THRESHOLD,
            unknown_threshold=THRESHOLD_REJECT,
            cohort_enabled=COHORT_ENABLED,
            cohort_z_threshold=COHORT_Z_THRESHOLD
        )
        
        def id_to_name_fn(valid_ids):
            if not valid_ids:
                return {}
            placeholders = ",".join("?" * len(valid_ids))
            rows = self._conn.execute(
                f"SELECT id, name FROM embeddings WHERE id IN ({placeholders})", tuple(valid_ids)
            ).fetchall()
            return {r[0]: r[1] for r in rows}
        
        return engine.match(
            query_emb, self._index, id_to_name_fn,
            self._proto_matrix, self._proto_names
        )

    def get_users(self):
        """Get dict {name: embedding_count}."""
        rows = self._conn.execute(
            "SELECT name, COUNT(*) FROM embeddings GROUP BY name"
        ).fetchall()
        return {n: c for n, c in rows}

    def get_prototype(self, name):
        """Get prototype vector for a user."""
        return self._prototypes.get(name)

    def has_prototype(self, name):
        """Check if user has a prototype."""
        return name in self._prototypes

    @property
    def total(self):
        return self._index.ntotal

    def close(self):
        self._conn.close()
