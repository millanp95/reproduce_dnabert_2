import numpy as np
import torch
import transformers

import webdataset as wds
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, max_len=512)


tokenized_seq = tokenizer("ACGT", padding='max_length')['input_ids']
print(len(tokenized_seq))

import io
import os
from pathlib import Path
import numpy as np
import webdataset as wds
from tqdm import tqdm
from multiprocessing import Pool
import os

# Suppress Hugging Face tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def tokenize(seq):
    tokens, _, att_mask = tokenizer(seq, padding='max_length').values()
    tokens_np = np.array(tokens).astype(np.uint16)
    att_mask_np = np.array(att_mask).astype(np.uint8)
    return tokens_np, att_mask_np

def npy_to_bytes(arr):
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()

def process_line(args):
    seq, split_name, idx = args
    tokens, mask = tokenize(seq.strip())
    key = f"{split_name}-{idx:09d}"
    return {
        "__key__": key,
        "tokens": npy_to_bytes(tokens),
        "attention_mask": npy_to_bytes(mask),
    }

def write_shards(input_file, out_dir, split_name, target_shard_size_bytes=1e9, max_samples_per_shard=300000, num_workers=8):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pattern = os.path.join(out_dir, f"{split_name}-%06d.tar")
    sink = wds.ShardWriter(pattern, maxsize=target_shard_size_bytes, maxcount=max_samples_per_shard)

    with open(input_file) as f, Pool(num_workers) as pool, tqdm(desc=f"Writing {split_name} shards") as pbar:
        for idx, line in enumerate(f):
            args = (line, split_name, idx)
            record = pool.apply(process_line, (args,))
            sink.write(record)
            pbar.update(1)

    sink.close()
    print(f"Wrote {idx + 1} samples to shards at {out_dir}")

# Get the number of logical CPUs in the system
cpu_count = os.cpu_count()

print(f"Number of available CPU cores: {cpu_count}")

write_shards("data/dev.txt", "shards", "dev")
write_shards("data/train.txt", "shards", "train",
max_samples_per_shard=600000, num_workers=8)
