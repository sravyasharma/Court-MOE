#!/usr/bin/env python3
import os
import torch
from collections import defaultdict

base_dir = "/home/infodna/Court-MOE"
input_path = os.path.join(base_dir, "encoded_output_final", "encoded_chunks_all.pth")
output_dir = os.path.join(base_dir, "encoded_output_final", "by_court")


os.makedirs(output_dir, exist_ok=True)


court_map = {
    0: "SupremeCourt",
    1: "HighCourt",
    2: "DistrictCourt",
    3: "TribunalCourt",
    4: "DailyOrderCourt"
}

print(f"📂 Loading encoded file: {input_path}")
data = torch.load(input_path, map_location="cpu")
print(f"✅ Loaded {len(data):,} total records")


court_groups = defaultdict(list)


for record in data:
    court_idx = record.get("court_type_idx", 4)
    court_name = court_map.get(court_idx, "DailyOrderCourt")
    court_groups[court_name].append(record)

for court_name, records in court_groups.items():
    out_path = os.path.join(output_dir, f"{court_name}_embeddings.pth")
    torch.save(records, out_path)
    print(f"💾 Saved {len(records):,} → {out_path}")


print("\n📊 Segregation Summary:")
for k, v in court_groups.items():
    print(f"  🏛 {k:<15} → {len(v):,} records")

print("\n✅ Done — segregated files saved under:")
print(f"   {output_dir}")
