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
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "supabase"])
    from supabase import create_client, Client

from config import DB_DIR, TOP_K


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

    def _auto_setup_tables(self):
        """Auto-create tables on Supabase if they don't exist."""
        import httpx

        print("  [DB] Checking Supabase tables...")

        # Use Supabase REST SQL endpoint (via pg-meta or rpc)
        # Method 1: Try using postgrest to check if tables exist
        try:
            # Quick test: try to select from face_embeddings
            self.supabase.table("face_embeddings").select("id").limit(1).execute()
            self.supabase.table("attendance_logs").select("id").limit(1).execute()
            print("  [DB] ✓ Tables already exist!")
            return
        except Exception:
            pass

        # Method 2: Create tables via Supabase Management API (SQL)
        print("  [DB] Tables not found. Creating via SQL...")
        
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
                print("  [DB] ✓ Tables created successfully via RPC!")
                return
        except Exception:
            pass

        # Method 3: If RPC doesn't work, print instructions
        print("  [DB] ⚠ Could not auto-create tables.")
        print("  [DB] Please create tables manually:")
        print("  " + "=" * 50)
        print("  1. Go to: https://supabase.com/dashboard")
        print("  2. Open your project → SQL Editor")
        print("  3. Run the SQL from: supabase_schema.sql")
        print("  " + "=" * 50)
        print("  [DB] Or run this command:")
        print(f'  curl -X POST "{self._url}/rest/v1/rpc/exec_sql" \\')
        print(f'    -H "apikey: {self._key[:20]}..." \\')
        print(f'    -H "Authorization: Bearer {self._key[:20]}..." \\')
        print('    -H "Content-Type: application/json" \\')
        print('    -d \'{"query": "CREATE TABLE IF NOT EXISTS ..."}\'')
        print()

    def _save(self):
        faiss.write_index(self._index, self._idx_path)
        with open(self._map_path, "w") as f:
            json.dump(self._id_map, f)

    def sync_from_supabase(self):
        """Download all embeddings from Supabase and rebuild FAISS index.
        Uses batch FAISS operations for speed.
        """
        import time
        t0 = time.time()

        # Fetch all rows (paginated if needed)
        result = self.supabase.table("face_embeddings").select("*").execute()
        rows = result.data

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
        print(f"  [SYNC] {len(rows)} embeddings synced in {elapsed:.2f}s")

    def add_user(self, name, embeddings, scores=None):
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
        elapsed = time.time() - t0
        print(f"  [ADD] {name}: {len(all_inserted)} embeddings in {elapsed:.2f}s")

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

    def match(self, query_emb):
        """Find best matching user via FAISS nearest neighbor."""
        if self._index.ntotal == 0:
            return "Unknown", 0.0

        k = min(TOP_K * 5, self._index.ntotal)
        scores, ids = self._index.search(
            np.array([query_emb], dtype=np.float32), k
        )

        user_scores = defaultdict(list)
        for score, eid in zip(scores[0], ids[0]):
            if eid == -1:
                continue
            info = self._id_map.get(str(int(eid)))
            if info:
                user_scores[info["name"]].append(float(score))

        best_name, best_score = "Unknown", 0.0
        for name, sc_list in user_scores.items():
            avg = np.mean(sorted(sc_list, reverse=True)[:TOP_K])
            if avg > best_score:
                best_score = avg
                best_name = name

        return best_name, best_score

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
