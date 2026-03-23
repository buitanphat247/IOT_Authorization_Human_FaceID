"""
PgVectorDatabase v5.2 - Supabase pgvector cloud-native matching (BUG-03 fix).

Instead of downloading ALL embeddings to local RAM for FAISS search,
this module queries Supabase pgvector directly for Top-K nearest neighbors.

Benefits:
  - No local FAISS index needed → saves 100s MB of RAM
  - Scales to millions of embeddings without memory issues
  - Single source of truth (no sync needed)
  
Requirements:
  1. Enable pgvector extension in Supabase:
     CREATE EXTENSION IF NOT EXISTS vector;
  2. Create the table with vector column:
     ALTER TABLE face_embeddings ADD COLUMN embedding_vec vector(512);
  3. Create HNSW index for fast search:
     CREATE INDEX ON face_embeddings USING hnsw (embedding_vec vector_cosine_ops);

Usage:
    from pgvector_db import PgVectorDatabase
    db = PgVectorDatabase(SUPABASE_URL, SUPABASE_KEY)
    name, score = db.match(query_emb)
"""

import os
import numpy as np
from collections import defaultdict
from datetime import datetime
from logger import get_logger

logger = get_logger("pgvector_db")

try:
    from supabase import create_client, Client
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "supabase"])
    from supabase import create_client, Client

from config import DB_DIR, TOP_K


class PgVectorDatabase:
    """Cloud-native vector search using Supabase pgvector.
    
    No local FAISS needed. All matching is done server-side.
    Requires pgvector extension enabled on Supabase.
    """

    _SETUP_SQL = """
    -- Enable pgvector extension
    CREATE EXTENSION IF NOT EXISTS vector;
    
    -- Add vector column if not exists
    DO $$ BEGIN
        ALTER TABLE face_embeddings ADD COLUMN embedding_vec vector(512);
    EXCEPTION
        WHEN duplicate_column THEN NULL;
    END $$;
    
    -- Create HNSW index for fast cosine search
    CREATE INDEX IF NOT EXISTS idx_face_embeddings_vec 
        ON face_embeddings USING hnsw (embedding_vec vector_cosine_ops);
    
    -- RPC function: vector search
    CREATE OR REPLACE FUNCTION match_face(
        query_embedding vector(512),
        match_count int DEFAULT 15
    ) RETURNS TABLE (
        id bigint,
        name text,
        similarity float
    ) LANGUAGE plpgsql AS $$
    BEGIN
        RETURN QUERY
        SELECT 
            fe.id,
            fe.name,
            1 - (fe.embedding_vec <=> query_embedding) AS similarity
        FROM face_embeddings fe
        WHERE fe.embedding_vec IS NOT NULL
        ORDER BY fe.embedding_vec <=> query_embedding
        LIMIT match_count;
    END;
    $$;
    """

    def __init__(self, supabase_url: str, supabase_key: str, db_dir=DB_DIR):
        os.makedirs(db_dir, exist_ok=True)
        self._dim = 512
        self._url = supabase_url
        self._key = supabase_key

        self.supabase: Client = create_client(supabase_url, supabase_key)

        # Prototype store (still local for fast O(1) matching)
        self._proto_path = os.path.join(db_dir, "prototypes.npy")
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

        # Count total vectors
        self._total = 0
        try:
            result = self.supabase.table("face_embeddings").select("id", count="exact").execute()
            self._total = result.count or 0
        except Exception:
            pass

        logger.info(f"PgVectorDatabase initialized ({self._total} vectors)")

    def _update_proto_cache(self):
        if self._prototypes:
            self._proto_matrix = np.array(list(self._prototypes.values()), dtype=np.float32)
            self._proto_names = list(self._prototypes.keys())
        else:
            self._proto_matrix = None
            self._proto_names = []

    def _save_prototypes(self):
        np.save(self._proto_path, self._prototypes)

    def match(self, query_emb):
        """Match using pgvector RPC (server-side vector search).
        
        Falls back to prototype-only matching if pgvector RPC fails.
        """
        from config import PROTOTYPE_ENABLED, PROTOTYPE_WEIGHT

        # --- Strategy 1: Prototype matching (local, instant) ---
        proto_name, proto_score = "Unknown", 0.0
        if PROTOTYPE_ENABLED and self._proto_matrix is not None and len(self._proto_names) > 0:
            query = np.array(query_emb, dtype=np.float32)
            sims = self._proto_matrix @ query
            best_idx = int(np.argmax(sims))
            proto_name = self._proto_names[best_idx]
            proto_score = float(sims[best_idx])

        # --- Strategy 2: pgvector RPC search (cloud, scalable) ---
        topk_name, topk_score = "Unknown", 0.0
        try:
            emb_list = query_emb.tolist() if hasattr(query_emb, 'tolist') else list(query_emb)
            result = self.supabase.rpc("match_face", {
                "query_embedding": emb_list,
                "match_count": TOP_K * 5
            }).execute()

            if result.data:
                user_scores = defaultdict(list)
                for row in result.data:
                    user_scores[row["name"]].append(float(row["similarity"]))

                for name, sc_list in user_scores.items():
                    avg = np.mean(sorted(sc_list, reverse=True)[:TOP_K])
                    if avg > topk_score:
                        topk_score = avg
                        topk_name = name
        except Exception as e:
            logger.warning(f"pgvector RPC failed: {e}. Using prototype-only matching.")

        # --- Fusion ---
        if PROTOTYPE_ENABLED and proto_name != "Unknown":
            w = PROTOTYPE_WEIGHT
            if proto_name == topk_name:
                final_score = w * proto_score + (1 - w) * topk_score
                final_name = proto_name
            else:
                if proto_score >= topk_score:
                    final_score = proto_score
                    final_name = proto_name
                else:
                    final_score = topk_score
                    final_name = topk_name
        else:
            final_score = topk_score
            final_name = topk_name

        return final_name, final_score

    def add_user(self, name, embeddings, scores=None, prototype=None):
        """Add user with pgvector-compatible embeddings."""
        import time
        t0 = time.time()

        self.remove_user(name)

        rows_to_insert = []
        now = datetime.utcnow().isoformat()

        for i, emb in enumerate(embeddings):
            s = scores[i] if scores else 0
            emb_list = emb.tolist() if isinstance(emb, np.ndarray) else emb
            rows_to_insert.append({
                "name": name,
                "embedding": emb_list,
                "embedding_vec": emb_list,  # pgvector column
                "quality_score": float(s),
                "created_at": now
            })

        # Batch insert
        BATCH_SIZE = 50
        for batch_start in range(0, len(rows_to_insert), BATCH_SIZE):
            batch = rows_to_insert[batch_start:batch_start + BATCH_SIZE]
            self.supabase.table("face_embeddings").insert(batch).execute()

        # Update prototype
        if prototype is not None:
            self._prototypes[name] = np.array(prototype, dtype=np.float32)
        elif embeddings:
            embs = np.array(embeddings, dtype=np.float32)
            proto = embs.mean(axis=0)
            proto = proto / np.linalg.norm(proto)
            self._prototypes[name] = proto

        self._save_prototypes()
        self._update_proto_cache()
        self._total += len(rows_to_insert)

        elapsed = time.time() - t0
        logger.info(f"ADD: {name}: {len(rows_to_insert)} embeddings in {elapsed:.2f}s (pgvector)")

    def remove_user(self, name):
        """Remove user from Supabase."""
        result = self.supabase.table("face_embeddings").delete().eq("name", name).execute()
        removed = len(result.data) if result.data else 0
        self._total = max(0, self._total - removed)

        if name in self._prototypes:
            del self._prototypes[name]
            self._save_prototypes()
            self._update_proto_cache()

    def get_users(self):
        result = self.supabase.table("face_embeddings").select("name").execute()
        user_counts = defaultdict(int)
        for row in result.data:
            user_counts[row["name"]] += 1
        return dict(user_counts)

    def get_attendance_logs(self, limit=50):
        result = self.supabase.table("attendance_logs") \
            .select("*").order("created_at", desc=True).limit(limit).execute()
        return result.data

    def log_attendance(self, name, score, status="PRESENT"):
        self.supabase.table("attendance_logs").insert({
            "name": name, "score": float(score),
            "status": status, "created_at": datetime.utcnow().isoformat()
        }).execute()

    @property
    def total(self):
        return self._total

    def close(self):
        pass
