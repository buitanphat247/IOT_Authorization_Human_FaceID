"""
SupabaseDatabase - Supabase + FAISS storage.
Replaces local SQLite with Supabase for cloud-based metadata storage.
FAISS still runs locally for fast vector search.
"""

import os
import json
import numpy as np
import faiss
from collections import defaultdict
from datetime import datetime

try:
    from supabase import create_client, Client
except ImportError:
    raise ImportError(
        "Missing 'supabase' package. Run: pip install supabase"
    )

from config import DB_DIR, TOP_K
from logger import get_logger

logger = get_logger("supabase_db")


class SupabaseDatabase:
    """Combined FAISS vector index + Supabase metadata store."""

    # SQL to auto-create tables
    _SETUP_SQL = """
    CREATE TABLE IF NOT EXISTS face_embeddings (
        id            BIGSERIAL PRIMARY KEY,
        name          TEXT NOT NULL,
        embedding     FLOAT8[] NOT NULL,
        quality_score FLOAT8 DEFAULT 0,
        created_at    TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_face_embeddings_name ON face_embeddings(name);

    CREATE TABLE IF NOT EXISTS attendance_logs (
        id          BIGSERIAL PRIMARY KEY,
        name        TEXT NOT NULL,
        score       FLOAT8 NOT NULL,
        status      TEXT DEFAULT 'PRESENT',
        created_at  TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_attendance_logs_time ON attendance_logs(created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_attendance_logs_name ON attendance_logs(name);
    """

    def __init__(self, supabase_url: str, supabase_key: str, db_dir=DB_DIR):
        os.makedirs(db_dir, exist_ok=True)
        self._idx_path = os.path.join(db_dir, "faiss.index")
        self._map_path = os.path.join(db_dir, "id_map.json")
        self._dim = 512
        self._url = supabase_url
        self._key = supabase_key

        # Supabase client
        self.supabase: Client = create_client(supabase_url, supabase_key)

        # Auto-create tables on first run
        self._auto_setup_tables()

        # FAISS
        if os.path.exists(self._idx_path):
            self._index = faiss.read_index(self._idx_path)
        else:
            base = faiss.IndexFlatIP(self._dim)
            self._index = faiss.IndexIDMap2(base)

        # Local ID map (faiss_id -> supabase_row_id) for syncing
        self._id_map = {}
        if os.path.exists(self._map_path):
            with open(self._map_path, "r") as f:
                self._id_map = json.load(f)
        
        self._next_id = max([int(k) for k in self._id_map.keys()], default=0) + 1
        
        # Prototype store: { name: numpy_512d }
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

    def _update_proto_cache(self):
        """Tạo ma trận numpy cache cho prototypes."""
        if self._prototypes:
            self._proto_matrix = np.array(list(self._prototypes.values()), dtype=np.float32)
            self._proto_names = list(self._prototypes.keys())
        else:
            self._proto_matrix = None
            self._proto_names = []

    def _save_prototypes(self):
        np.save(self._proto_path, self._prototypes)

    def _auto_setup_tables(self):
        """Auto-create tables on Supabase if they don't exist."""
        import httpx

        logger.info("Checking Supabase tables...")

        # Use Supabase REST SQL endpoint (via pg-meta or rpc)
        # Method 1: Try using postgrest to check if tables exist
        try:
            # Quick test: try to select from face_embeddings
            self.supabase.table("face_embeddings").select("id").limit(1).execute()
            self.supabase.table("attendance_logs").select("id").limit(1).execute()
            logger.info("✓ Tables already exist!")
            return
        except Exception:
            pass

        # Method 2: Create tables via Supabase Management API (SQL)
        logger.info("Tables not found. Creating via SQL...")
        
        try:
            # Use the Supabase SQL endpoint
            headers = {
                "apikey": self._key,
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json"
            }
            
            # Execute SQL via pg-meta REST endpoint
            sql_url = f"{self._url}/rest/v1/rpc/exec_sql"
            
            # Try RPC method first
            resp = httpx.post(sql_url, json={"query": self._SETUP_SQL}, headers=headers, timeout=30)
            
            if resp.status_code in (200, 201):
                logger.info("✓ Tables created successfully via RPC!")
                return
        except Exception:
            pass

        # Method 3: If RPC doesn't work, print instructions
        logger.warning("Could not auto-create tables.")
        logger.warning("Please create tables manually:")
        logger.warning("="*50)
        logger.warning("1. Go to: https://supabase.com/dashboard")
        logger.warning("2. Open your project → SQL Editor")
        logger.warning("3. Run the SQL from: supabase_schema.sql")
        logger.warning("="*50)

    def _save(self):
        faiss.write_index(self._index, self._idx_path)
        with open(self._map_path, "w") as f:
            json.dump(self._id_map, f)

    def sync_from_supabase(self):
        """Download all embeddings from Supabase and rebuild FAISS index.
        
        BUG-03 mitigation:
        - Paginated fetch (1000 rows/batch) to avoid Supabase timeout
        - Skip re-sync if FAISS already matches cloud count
        - RAM usage warning for large datasets
        """
        import time
        t0 = time.time()

        # Quick count check — skip sync if FAISS already up-to-date
        try:
            count_result = self.supabase.table("face_embeddings") \
                .select("id", count="exact").execute()
            cloud_count = count_result.count if count_result.count else len(count_result.data)
        except Exception:
            cloud_count = None
        
        if cloud_count is not None and self._index.ntotal == cloud_count and cloud_count > 0:
            logger.info(f"SYNC: FAISS already has {cloud_count} vectors = cloud count. Skipping full sync.")
            return

        # RAM estimation warning
        if cloud_count and cloud_count > 5000:
            est_ram_mb = cloud_count * self._dim * 4 / (1024 * 1024)
            logger.warning(f"SYNC WARNING: {cloud_count} embeddings will use ~{est_ram_mb:.0f} MB RAM in FAISS.")
            logger.warning("Consider switching to DB_BACKEND=pgvector for large datasets.")

        # Paginated fetch (1000 rows per batch)
        PAGE_SIZE = 1000
        rows = []
        offset = 0
        while True:
            batch = self.supabase.table("face_embeddings") \
                .select("*") \
                .range(offset, offset + PAGE_SIZE - 1) \
                .execute()
            if not batch.data:
                break
            rows.extend(batch.data)
            if len(batch.data) < PAGE_SIZE:
                break
            offset += PAGE_SIZE

        # Rebuild FAISS with batch add
        base = faiss.IndexFlatIP(self._dim)
        self._index = faiss.IndexIDMap2(base)
        self._id_map = {}
        self._next_id = 1

        if rows:
            # Batch: build arrays first, then add all at once
            all_vecs = np.zeros((len(rows), self._dim), dtype=np.float32)
            all_ids = np.zeros(len(rows), dtype=np.int64)

            for i, row in enumerate(rows):
                all_vecs[i] = np.array(row["embedding"], dtype=np.float32)
                faiss_id = self._next_id
                self._next_id += 1
                all_ids[i] = faiss_id
                self._id_map[str(faiss_id)] = {
                    "supabase_id": row["id"],
                    "name": row["name"]
                }

            # Single batch FAISS add (much faster than one-by-one)
            self._index.add_with_ids(all_vecs, all_ids)

        self._save()
        elapsed = time.time() - t0
        logger.info(f"SYNC: {len(rows)} embeddings synced in {elapsed:.2f}s")

    def add_user(self, name, embeddings, scores=None, prototype=None):
        """Add user with embeddings to Supabase + FAISS.
        Uses BATCH INSERT for Supabase (1 API call instead of N).
        """
        import time
        t0 = time.time()

        self.remove_user(name)

        # Prepare all rows for batch insert
        rows_to_insert = []
        emb_lists = []
        now = datetime.utcnow().isoformat()

        for i, emb in enumerate(embeddings):
            s = scores[i] if scores else 0
            emb_list = emb.tolist() if isinstance(emb, np.ndarray) else emb
            emb_lists.append(emb_list)
            rows_to_insert.append({
                "name": name,
                "embedding": emb_list,
                "quality_score": float(s),
                "created_at": now
            })

        # BATCH INSERT: one API call for all rows (huge speedup!)
        BATCH_SIZE = 50  # Supabase handles up to ~1000 rows per call
        all_inserted = []

        for batch_start in range(0, len(rows_to_insert), BATCH_SIZE):
            batch = rows_to_insert[batch_start:batch_start + BATCH_SIZE]
            result = self.supabase.table("face_embeddings").insert(batch).execute()
            all_inserted.extend(result.data)

        # Batch FAISS add
        if all_inserted:
            n = len(all_inserted)
            all_vecs = np.zeros((n, self._dim), dtype=np.float32)
            all_ids = np.zeros(n, dtype=np.int64)

            for i, row in enumerate(all_inserted):
                all_vecs[i] = np.array(emb_lists[i], dtype=np.float32)
                faiss_id = self._next_id
                self._next_id += 1
                all_ids[i] = faiss_id
                self._id_map[str(faiss_id)] = {
                    "supabase_id": row["id"],
                    "name": name
                }

            self._index.add_with_ids(all_vecs, all_ids)

        self._save()
        
        # Cập nhật prototype cục bộ
        if prototype is not None:
            self._prototypes[name] = np.array(prototype, dtype=np.float32)
        elif embeddings:
            embs = np.array(embeddings, dtype=np.float32)
            proto = embs.mean(axis=0)
            proto = proto / np.linalg.norm(proto)
            self._prototypes[name] = proto
            
        self._save_prototypes()
        self._update_proto_cache()
        
        elapsed = time.time() - t0
        logger.info(f"ADD: {name}: {len(all_inserted)} embeddings in {elapsed:.2f}s")

    def remove_user(self, name):
        """Remove all embeddings for a user from Supabase + FAISS."""
        # Remove from Supabase
        self.supabase.table("face_embeddings").delete().eq("name", name).execute()

        # Remove from FAISS
        ids_to_remove = []
        for fid, info in list(self._id_map.items()):
            if info["name"] == name:
                ids_to_remove.append(int(fid))
                del self._id_map[fid]

        if ids_to_remove:
            self._index.remove_ids(np.array(ids_to_remove, dtype=np.int64))
            self._save()

        # Remove prototype
        if name in self._prototypes:
            del self._prototypes[name]
            self._save_prototypes()
            self._update_proto_cache()

    def match(self, query_emb):
        """Find best matching user via dual-strategy matching.
        Delegates to shared MatchingEngine (BUG-09 fix).
        """
        from matching import MatchingEngine
        from config import PROTOTYPE_ENABLED, PROTOTYPE_WEIGHT
        
        engine = MatchingEngine(
            top_k=TOP_K,
            proto_weight=PROTOTYPE_WEIGHT,
            proto_enabled=PROTOTYPE_ENABLED
        )
        
        def id_to_name_fn(valid_ids):
            """Resolve FAISS IDs to user names via local id_map."""
            if not valid_ids:
                return {}
            result = {}
            for vid in valid_ids:
                info = self._id_map.get(str(vid))
                if info:
                    result[vid] = info["name"]
            return result
        
        return engine.match(
            query_emb, self._index, id_to_name_fn,
            self._proto_matrix, self._proto_names
        )

    def get_prototype(self, name):
        """Get prototype vector for a user."""
        return self._prototypes.get(name)

    def has_prototype(self, name):
        """Check if user has a prototype."""
        return name in self._prototypes

    def get_users(self):
        """Get dict {name: embedding_count} from Supabase."""
        result = self.supabase.table("face_embeddings") \
            .select("name") \
            .execute()
        
        user_counts = defaultdict(int)
        for row in result.data:
            user_counts[row["name"]] += 1
        return dict(user_counts)

    def get_attendance_logs(self, limit=50):
        """Get recent attendance logs from Supabase."""
        result = self.supabase.table("attendance_logs") \
            .select("*") \
            .order("created_at", desc=True) \
            .limit(limit) \
            .execute()
        return result.data

    def log_attendance(self, name, score, status="PRESENT"):
        """Log attendance event to Supabase."""
        self.supabase.table("attendance_logs").insert({
            "name": name,
            "score": float(score),
            "status": status,
            "created_at": datetime.utcnow().isoformat()
        }).execute()

    @property
    def total(self):
        return self._index.ntotal

    def close(self):
        pass  # Supabase uses HTTP, no connection to close
