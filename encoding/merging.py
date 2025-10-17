import torch
from collections import defaultdict
import numpy as np

path = "/home/infodna/Court-MOE/encoded_output_final/encoded_chunks_all.pth"
out_path = "/home/infodna/Court-MOE/encoded_output_final/aggregated_doc_embeddings.pth"

data = torch.load(path, map_location="cpu")
print(f"Loaded {len(data):,} chunk embeddings")

case_groups = defaultdict(list)
meta_info = {}

for rec in data:
    case_id = rec["case_id"]
    case_groups[case_id].append(rec["embeddings"])
    meta_info[case_id] = {
        "label": rec["label"],
        "court_type_idx": rec["court_type_idx"]
    }

aggregated = []
for case_id, embs in case_groups.items():
    mean_vec = np.mean(embs, axis=0).tolist()
    aggregated.append({
        "case_id": case_id,
        "label": meta_info[case_id]["label"],
        "court_type_idx": meta_info[case_id]["court_type_idx"],
        "embedding": mean_vec
    })

torch.save(aggregated, out_path)
print(f"✅ Saved {len(aggregated):,} aggregated document embeddings → {out_path}")
