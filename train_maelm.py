import os
import time
import math
import argparse
import random
import torch
import torch.nn as nn
import torch.distributed as dist
import webdataset as wds
import numpy as np
import monitor # Assuming monitor.py is available
from transformers import AutoTokenizer, BertConfig, get_cosine_schedule_with_warmup
from torch.nn.parallel import DistributedDataParallel as DDP

# Import the new MAELM model
try:
    from maelm_model import MAELMModel
except ImportError:
    print("Warning: maelm_model.py not found. Please ensure the file exists.")
    MAELMModel = None

def parse_args():
    parser = argparse.ArgumentParser(description="Train MAELM Model")
    # DDP args
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", -1)))
    
    # Model/Training args
    parser.add_argument("--batch-size", type=int, default=8, help="Per device batch size")
    parser.add_argument("--total-batch-size", type=int, default=4096, help="Global batch size")
    parser.add_argument("--max-steps", type=int, default=500000)
    parser.add_argument("--warmup-steps", type=int, default=30000)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--checkpoint-interval", type=int, default=2000)
    
    # Paths
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="log")
    parser.add_argument("--train-shards-pattern", type=str, default="shards/train-*.tar")
    parser.add_argument("--mask-ratio", type=float, default=0.15)
    
    # Fine-tuning / Monitoring
    parser.add_argument("--finetune-data-path", type=str, default=None, help="Path to data for fine-tuning evaluation")
    
    # Compatibility args 
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--alibi-starting-size", type=int, default=512)
    parser.add_argument("--no-compile", action="store_true")

    return parser.parse_known_args()[0]

def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        return rank, local_rank, world_size
    else:
        return 0, 0, 1

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def get_dataloader(pattern, batch_size, is_train=True):
    dataset = wds.WebDataset(pattern, resampled=is_train, shardshuffle=is_train)
    if is_train:
        dataset = dataset.shuffle(1000)
    dataset = dataset.decode() # Assuming safe
    dataset = dataset.to_tuple("input_ids.npy", "attention_mask.npy")
    dataset = dataset.batched(batch_size)
    loader = wds.WebLoader(dataset, batch_size=None, num_workers=4)
    if is_train:
        loader = loader.unbatched().shuffle(1000).batched(batch_size)
    return loader

def process_batch_mlm(input_ids, attention_mask, tokenizer, mask_ratio=0.15):
    labels = input_ids.clone()
    probability_matrix = torch.full(labels.shape, mask_ratio)
    
    special_tokens_mask = torch.zeros_like(labels, dtype=torch.bool)
    if tokenizer.pad_token_id is not None:
        special_tokens_mask |= (labels == tokenizer.pad_token_id)
    if tokenizer.cls_token_id is not None:
        special_tokens_mask |= (labels == tokenizer.cls_token_id)
    if tokenizer.sep_token_id is not None:
        special_tokens_mask |= (labels == tokenizer.sep_token_id)
        
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100 

    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    input_ids[indices_replaced] = tokenizer.mask_token_id

    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    input_ids[indices_random] = random_words[indices_random]
    
    return input_ids, labels

def save_checkpoint(model, optimizer, step, args, rank):
    if rank != 0: return
    
    step_dir = os.path.join(args.checkpoint_dir, f"step_{step}")
    os.makedirs(step_dir, exist_ok=True)
    
    # Save Model Weights (raw)
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(), os.path.join(step_dir, "pytorch_model.bin"))
    
    # Save Config
    model_to_save.encoder.config.save_pretrained(step_dir)

    # Save Optimizer & Args (for Resuming)
    train_state = {
        'optimizer': optimizer.state_dict(),
        'step': step,
        'args': args
    }
    torch.save(train_state, os.path.join(step_dir, "training_state.pt"))
    
    # Symlink latest
    latest_link = os.path.join(args.checkpoint_dir, "latest")
    if os.path.exists(latest_link):
        os.remove(latest_link)
    os.symlink(os.path.abspath(step_dir), latest_link)

def main():
    args = parse_args()
    rank, local_rank, world_size = setup_ddp()
    
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    acc_steps = max(1, args.total_batch_size // (args.batch_size * world_size))

    if rank == 0:
        print(f"MAELM Training | Batch: {args.total_batch_size} | Accum: {acc_steps} | Finetune: {args.finetune_data_path}")
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

    config = BertConfig(**{
        "vocab_size": 4096, 
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "max_position_embeddings": args.max_seq_length,
        "type_vocab_size": 2, 
        "_name_or_path": "zhihan1996/DNABERT-2-117M"
    })

    model = MAELMModel(config, config) 
    model.to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps
    )

    try:
        dataloader = get_dataloader(args.train_shards_pattern, args.batch_size, is_train=True)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    model.train()
    step = 0
    start_time = time.time()
    optimizer.zero_grad()

    try:
        for i, batch in enumerate(dataloader):
            step += 1
            
            if isinstance(batch, (tuple, list)):
                input_ids, attention_mask = batch[0], batch[1]
            elif isinstance(batch, dict):
                input_ids = batch.get('input_ids.npy', batch.get('input_ids'))
                attention_mask = batch.get('attention_mask.npy', batch.get('attention_mask'))
            
            input_ids = torch.as_tensor(input_ids).long().to(device)
            attention_mask = torch.as_tensor(attention_mask).long().to(device)
            
            input_ids, labels = process_batch_mlm(input_ids, attention_mask, tokenizer, args.mask_ratio)
            labels = labels.to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            loss = outputs.loss if hasattr(outputs, 'loss') else (outputs[0] if isinstance(outputs, (tuple, list)) else outputs)
            loss = loss / acc_steps
            loss.backward()
            
            if step % acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
            if step % 100 == 0 and rank == 0:
                elapsed = time.time() - start_time
                if elapsed == 0: elapsed = 1e-5
                tok_per_sec = (args.total_batch_size * args.max_seq_length * 100) / elapsed
                lr = scheduler.get_last_lr()[0]
                print(f"Step {step} | Loss: {loss.item()*acc_steps:.4f} | LR: {lr:.2e} | Tok/s: {tok_per_sec:.0f}")
                with open(os.path.join(args.log_dir, "log.txt"), "a") as f:
                    f.write(f"{step} train {loss.item()*acc_steps:.4f} lr {lr:.2e} tok/s {tok_per_sec:.0f}\n")
                start_time = time.time()
                
            if step % args.checkpoint_interval == 0 and rank == 0:
                save_checkpoint(model, optimizer, step, args, rank)
                
                # --- MONITORING TRIGGER ---
                # Check finetune data path and valid monitor module
                if args.finetune_data_path and monitor:
                    print(f"Step {step}: Triggering fine-tuning monitoring...")
                    finetune_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "finetune.py")
                    
                    # Directory where model was saved
                    step_dir = os.path.join(args.checkpoint_dir, f"step_{step}")
                    history_path = os.path.join(args.log_dir, "finetune_history.csv")
                    
                    # Call monitor.run_finetune with NEW signature
                    metrics = monitor.run_finetune(
                        finetune_script, 
                        step_dir, 
                        args.finetune_data_path, 
                        args.log_dir, 
                        step,
                        model_type="maelm"
                    )
                    
                    if metrics:
                        print(f"Step {step} finetune metrics: {metrics}")
                        monitor.update_finetune_history(history_path, step, metrics)
                        monitor.plot_curves(os.path.join(args.log_dir, "log.txt"), history_path, args.log_dir)
                    else:
                        print(f"Step {step} finetune returned no metrics.")
            
            if step >= args.max_steps:
                break
                
    except Exception as e:
        print(f"Error during training on rank {rank}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_ddp()

if __name__ == "__main__":
    main()
