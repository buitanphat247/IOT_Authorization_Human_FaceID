"""Quick debug: check WHAT the matching engine returns for prototypes"""
import sys, os
sys.path.insert(0, r'e:\Workspace\detect\core')

import numpy as np

# Load prototypes
proto_path = r'e:\Workspace\detect\db\prototypes.npy'
if os.path.exists(proto_path):
    data = np.load(proto_path, allow_pickle=True).item()
    print(f"=== Prototypes: {len(data)} users ===")
    for name, proto in data.items():
        norm = np.linalg.norm(proto)
        print(f"  {name}: shape={proto.shape}, L2 norm={norm:.4f}")
        
    if len(data) > 0:
        names = list(data.keys())
        protos = np.array(list(data.values()), dtype=np.float32)
        
        print(f"\n=== Self-match (user vs own prototype): ===")
        for i, name in enumerate(names):
            sim = float(protos[i] @ protos[i])
            print(f"  {name} vs self: cosine={sim:.4f}")
        
        print(f"\n=== Random noise vs prototypes: ===")
        random_emb = np.random.randn(512).astype(np.float32)
        random_emb = random_emb / np.linalg.norm(random_emb)
        sims = protos @ random_emb
        for i, name in enumerate(names):
            print(f"  Random vs {name}: cosine={float(sims[i]):.4f}")
else:
    print(f"No prototypes found at {proto_path}")

# Check FAISS index
idx_path = r'e:\Workspace\detect\db\faiss.index'
if os.path.exists(idx_path):
    import faiss
    index = faiss.read_index(idx_path)
    print(f"\n=== FAISS Index ===")
    print(f"  Total vectors: {index.ntotal}")
    print(f"  Dimension: {index.d}")
    
    # Search with random embedding
    random_emb = np.random.randn(512).astype(np.float32)
    random_emb = random_emb / np.linalg.norm(random_emb)
    scores, ids = index.search(np.array([random_emb], dtype=np.float32), min(5, index.ntotal))
    print(f"\n=== Random embedding vs FAISS (should be LOW ~0.0-0.3): ===")
    for i in range(len(ids[0])):
        if ids[0][i] != -1:
            print(f"  ID={ids[0][i]}, score={scores[0][i]:.4f}")
else:
    print(f"No FAISS index at {idx_path}")
