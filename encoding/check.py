#!/usr/bin/env python3
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


AGGREGATED_PATH = "/home/infodna/Court-MOE/encoded_output_final/aggregated_doc_embeddings.pth"

print(f"🔍 Loading aggregated embeddings from {AGGREGATED_PATH} ...")
data = torch.load(AGGREGATED_PATH, map_location="cpu")

print(f"✅ Loaded {len(data):,} aggregated records\n")


embedding_lengths = [len(rec["embedding"]) for rec in data]
unique_lengths = set(embedding_lengths)

if len(unique_lengths) == 1 and 768 in unique_lengths:
    print("✅ All embeddings have correct length: 768\n")
else:
    print(f"⚠️ Inconsistent embedding sizes found: {unique_lengths}\n")


court_counts = Counter(rec["court_type_idx"] for rec in data)
court_map = {
    0: "SupremeCourt",
    1: "HighCourt",
    2: "DistrictCourt",
    3: "TribunalCourt",
    4: "DailyOrderCourt"
}

print("📊 Court-wise Case Distribution:")
for court_idx, count in sorted(court_counts.items()):
    name = court_map.get(court_idx, f"Unknown ({court_idx})")
    print(f"   🏛 {name:<16} → {count:,} cases")


print("\n📈 Computing embedding variance per court...")

court_variances = {court_map[idx]: [] for idx in court_counts.keys()}

for rec in data:
    emb = np.array(rec["embedding"], dtype=np.float32)
    var = float(np.var(emb))
    court_variances[court_map[rec["court_type_idx"]]].append(var)

mean_vars = {court: np.mean(vars) for court, vars in court_variances.items()}

plt.figure(figsize=(8,5))
plt.bar(mean_vars.keys(), mean_vars.values(), color='skyblue')
plt.title("Mean Embedding Variance per Court")
plt.ylabel("Variance")
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig("/home/infodna/Court-MOE/encoded_output_final/embedding_variance_per_court.png")
plt.show()

print("\n✅ Verification complete.")
print("📁 Plot saved to: /home/infodna/Court-MOE/encoded_output_final/embedding_variance_per_court.png")
