#!/usr/bin/env python3
import os
import json
import logging
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch import amp


os.environ["CUDA_VISIBLE_DEVICES"] = "2"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("encode_gpu2_pakka")

def set_seed(seed: int = 42):
    """Ensure reproducibility across runs."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FlattenedChunkDataset(Dataset):
    """Each row = one chunk of a legal document."""
    def __init__(self, flattened):
        self.data = flattened
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "label": torch.tensor(item["label"], dtype=torch.long),
            "court_type_idx": torch.tensor(item["court_type_idx"], dtype=torch.long),
            "case_id": item["case_id"]
        }


def encode_tokenized_output(
    tokenized_path: str,
    output_dir: str,
    model_name: str = "nlpaueb/legal-bert-base-uncased",
    tokenizer_name: str = "/home/infodna/Court-MOE/tokenization/final_tokenizer",
    batch_size: int = 8,
    max_length: int = 512,
    max_samples: int | None = None,
):
    set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"🧠 Using device: {device}")

    # === Load tokenizer + model ===
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModel.from_pretrained(model_name)

    # Resize embeddings if tokenizer vocab is larger
    tok_len = len(tokenizer)
    emb_len = model.get_input_embeddings().weight.size(0)
    if tok_len > emb_len:
        model.resize_token_embeddings(tok_len)
        logger.info(f"Resized model embeddings → {tok_len} tokens (was {emb_len})")

    model.to(device)
    model.eval()

    logger.info(f"📥 Loading tokenized dataset from {tokenized_path}")
    with open(tokenized_path, "r", encoding="utf-8") as f:
        entries = [json.loads(line.strip()) for line in f if line.strip()]
    if max_samples:
        entries = entries[:max_samples]
        logger.info(f"⚙️ Test mode: using first {max_samples} entries")

    
    label_map = {"Rejected": 0, "Accepted": 1}
    court_map = {
        "SupremeCourt": 0,
        "HighCourt": 1,
        "DistrictCourt": 2,
        "TribunalCourt": 3,
        "DailyOrderCourt": 4,
    }

    
    flattened = []
    for entry in entries:
        orig_index = entry.get("orig_index", 0)
        label = label_map.get(entry.get("label", "Rejected"), 0)
        court_type = entry.get("court_type", "DailyOrderCourt")
        court_idx = court_map.get(court_type, 4)

        for chunk in entry.get("chunks", []):
            token_ids = chunk.get("token_ids", [])
            attention_mask = [1] * len(token_ids)

            # Pad/truncate
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
                attention_mask = attention_mask[:max_length]
            else:
                pad_len = max_length - len(token_ids)
                token_ids += [0] * pad_len
                attention_mask += [0] * pad_len

            flattened.append({
                "input_ids": token_ids,
                "attention_mask": attention_mask,
                "label": label,
                "court_type_idx": court_idx,
                "case_id": f"case_{orig_index}"
            })

    logger.info(f"📊 Total flattened chunks: {len(flattened):,}")
    if len(flattened) == 0:
        raise RuntimeError("❌ No valid chunks found in tokenized dataset!")

    dataset = FlattenedChunkDataset(flattened)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    all_records = []
    total_batches = len(loader)
    logger.info(f"🚀 Starting encoding: {total_batches:,} batches total...")

    os.makedirs(output_dir, exist_ok=True)

   
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Encoding", ncols=100)):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            try:
                with amp.autocast('cuda'):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    cls_emb = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()

                for i in range(cls_emb.shape[0]):
                    all_records.append({
                        "embeddings": cls_emb[i].tolist(),
                        "label": int(batch["label"][i]),
                        "court_type_idx": int(batch["court_type_idx"][i]),
                        "case_id": batch["case_id"][i]
                    })

            except Exception as e:
                logger.error(f"⚠️ Error at batch {batch_idx}: {e}")
                continue

           
            if (batch_idx + 1) % 10000 == 0:
                ckpt_path = os.path.join(output_dir, f"checkpoint_{batch_idx+1}.pth")
                torch.save(all_records, ckpt_path)
                logger.info(f"💾 Checkpoint saved: {ckpt_path} ({len(all_records)} records)")

    final_pth = os.path.join(output_dir, "encoded_chunks_all.pth")
    torch.save(all_records, final_pth)

    logger.info(f"✅ Saved {len(all_records):,} encoded chunks → {final_pth}")
    logger.info(f"📦 File size: {os.path.getsize(final_pth)/1e6:.2f} MB")


if __name__ == "__main__":
    BASE_DIR = "/home/infodna/Court-MOE"
    TOKENIZED_PATH = os.path.join(BASE_DIR, "tokenization", "Tokenized_output_pakka", "tokenized_multi_fixed_1760646787.jsonl")
    OUTPUT_DIR = os.path.join(BASE_DIR, "encoded_output_pakka")

    encode_tokenized_output(
        tokenized_path=TOKENIZED_PATH,
        output_dir=OUTPUT_DIR,
        model_name="nlpaueb/legal-bert-base-uncased",
        tokenizer_name="/home/infodna/Court-MOE/tokenization/final_tokenizer",
        batch_size=8,
        max_length=512,
    )
