import io
import os
import time
import math
import argparse
import random
import glob
import torch
import torch.nn as nn
import torch.distributed as dist
import webdataset as wds
import numpy as np
import warnings
import monitor
from transformers import AutoTokenizer, BertConfig, get_cosine_schedule_with_warmup
from torch.nn.parallel import DistributedDataParallel as DDP

# Import the new MAELM model
try:
    from maelm_model import MAELMModel
except ImportError:
    print("Warning: maelm_model.py not found. Please ensure the file exists.")
    MAELMModel = None

def get_num_cpu_available():
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        warnings.warn("Unable to determine number of available CPUs. Falling back to os.cpu_count().")
        return os.cpu_count()

def npy_decoder(data: bytes) -> torch.Tensor:
    return torch.from_numpy(np.load(io.BytesIO(data), allow_pickle=False))

def make_dataset(pattern: str, shuffle_buf: int = 10_000, resampled: bool = False, world_size: int = 1, rank: int = 0):
    # Create WebDataset with proper nodesplitter for distributed training
    if world_size > 1:
        # Use explicit nodesplitter for multi-GPU/multi-node training
        ds = wds.WebDataset(
            pattern, 
            resampled=resampled,
            nodesplitter=wds.shardlists.split_by_node
        )
    else:
        # Single GPU training
        ds = wds.WebDataset(pattern, resampled=resampled, shardshuffle=True)
    
    # Add epoch and length for continuous training
    if resampled:
        ds = ds.with_epoch(1000000)  # Large epoch size for continuous training
        ds = ds.with_length(1000000)  # Ensure dataset has sufficient length
    
    # Shuffle and decode
    ds = ds.shuffle(shuffle_buf)  # shuffle *within* the streaming buffer
    ds = ds.to_tuple("__key__", "tokens", "attention_mask").map_tuple(lambda x:x, npy_decoder, npy_decoder)
    return ds

def setup_data_loaders(batch_size, total_train_batch_size, train_shards_pattern, world_size, rank, device, workers=4):
    """Setup training data loader matching train.py logic"""
    assert total_train_batch_size % (batch_size * world_size) == 0
    grad_accum_steps = total_train_batch_size // (batch_size * world_size)
    
    if rank == 0:
        print(f"total desired batch size: {total_train_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
    
    # Setup datasets
    train_dataset_patterns = sorted(glob.glob(train_shards_pattern))
    if not train_dataset_patterns: 
        train_dataset_patterns = train_shards_pattern 
        
    train_dataset = make_dataset(train_dataset_patterns, 10000, resampled=True, 
                                world_size=world_size, rank=rank)
    
    # Use WebLoader for WebDataset
    cpu_workers = workers
    
    if world_size > 1:
        # For distributed training, use WebLoader
        dataloader_train = iter(wds.WebLoader(
            train_dataset.batched(batch_size, partial=False),
            num_workers=cpu_workers,
            batch_size=None 
        ))
    else:
        # Single GPU can still use regular DataLoader
        dl_train_kwargs = {
            "batch_size": batch_size,
            "drop_last": True,
            "sampler": None,
            "shuffle": False,
            "num_workers": cpu_workers,
            "pin_memory": device.type != "cpu"
        }
        dataloader_train = iter(torch.utils.data.DataLoader(train_dataset, **dl_train_kwargs))
    
    return dataloader_train, grad_accum_steps

def process_batch_mlm(batch, tokenizer, mask_ratio=0.15):
    """
    Process batch for masked language modeling.
    """
    _, input_ids, att_mask = batch
    
    # Ensure they are tensors
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.as_tensor(input_ids)
    if not isinstance(att_mask, torch.Tensor):
        att_mask = torch.as_tensor(att_mask)

    # Convert to appropriate types immediately
    input_ids = input_ids.long()
    att_mask = att_mask.long()
        
    # Validation
    if input_ids.size(1) > 512:
        input_ids = input_ids[:, :512]
        att_mask = att_mask[:, :512]
    
    # Check for invalid tokens
    if (input_ids >= 4096).any():
        input_ids = torch.clamp(input_ids, max=4095)
        
    targets = input_ids.clone()
    att_mask = att_mask.to(dtype=torch.float32)
    
    # 5 special tokens: CLS, PAD, SEP, UNK, MASK
    num_special_tokens = 5 
    mask_token_id = int(tokenizer.mask_token_id) if tokenizer.mask_token_id is not None else 4
    
    valid_token_mask = targets > num_special_tokens
    random_mask = torch.rand(targets.shape)
    input_maskout = random_mask < mask_ratio
    input_maskout &= valid_token_mask 
    
    masked_input = targets.detach().clone()
    masked_input.masked_fill_(input_maskout, mask_token_id)
    att_mask.masked_fill_(input_maskout, 0.0) 
    
    return masked_input, att_mask, targets

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
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loader workers")

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

def save_checkpoint(model, optimizer, step, args, rank):
    if rank != 0: return
    
    step_dir = os.path.join(args.checkpoint_dir, f"step_{step}")
    os.makedirs(step_dir, exist_ok=True)
    
    # Save Model Weights (raw)
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(), os.path.join(step_dir, "pytorch_model.bin"))
    
    # Save Config
    if hasattr(model_to_save, "encoder") and hasattr(model_to_save.encoder, "config"):
         model_to_save.encoder.config.save_pretrained(step_dir)
    elif hasattr(model_to_save, "config"):
         model_to_save.config.save_pretrained(step_dir)

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

    if rank == 0:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)

    # Data Loading
    train_loader, acc_steps = setup_data_loaders(
        args.batch_size, 
        args.total_batch_size, 
        args.train_shards_pattern, 
        world_size, 
        rank, 
        device,
        workers=args.num_workers
    )

    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

    config = BertConfig(**{
        "vocab_size": 4096, 
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "max_position_embeddings": args.max_seq_length,
        "type_vocab_size": 2, 
        "_name_or_path": "zhihan1996/DNABERT-2-117M",
        "alibi_starting_size": args.alibi_starting_size,
        "num_labels": 4096,
    })

    print("Initializing MAELMModel...")
    model = MAELMModel(config, config) 
    model.to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps
    )

    model.train()
    step = 0
    start_time = time.time()
    optimizer.zero_grad()

    try:
        if rank == 0:
            print(f"MAELM Training | Steps: {args.max_steps} | Accum: {acc_steps}")

        for step in range(0, args.max_steps):
            loss_accum = 0.0
            
            for micro_step in range(acc_steps):
                try:
                    batch = next(train_loader)
                except (StopIteration, Exception) as e:
                    if rank == 0:
                        print(f"DataLoader issue ({type(e).__name__}), recreating...")
                        if isinstance(e, (BrokenPipeError, ConnectionResetError)):
                             time.sleep(10) # Wait a bit before retrying if transport issue
                             
                    # Recreate loader
                    train_loader, _ = setup_data_loaders(
                        args.batch_size, args.total_batch_size, args.train_shards_pattern, world_size, rank, device
                    )
                    batch = next(train_loader)
                
                input_ids, attention_mask, targets = process_batch_mlm(batch, tokenizer, args.mask_ratio)
                
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                targets = targets.to(device)
                
                if world_size > 1:
                    model.require_backward_grad_sync = (micro_step == acc_steps - 1)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=targets)
                
                loss = outputs.loss if hasattr(outputs, 'loss') else (outputs[0] if isinstance(outputs, (tuple, list)) else outputs)
                loss = loss / acc_steps
                loss_accum += loss.detach()
                loss.backward()
            
            if world_size > 1:
                dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
                
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
                
            if step % 100 == 0 and rank == 0:
                elapsed = time.time() - start_time
                if elapsed == 0: elapsed = 1e-5
                seq_per_sec = (args.total_batch_size * args.max_seq_length) / elapsed # Using max_seq_length for approx token count or just batch size
                lr = scheduler.get_last_lr()[0]
                print(f"Step {step} | Loss: {loss_accum.item():.4f} | LR: {lr:.2e} | Tok/s: {seq_per_sec:.0f}")
                with open(os.path.join(args.log_dir, "log.txt"), "a") as f:
                    f.write(f"{step} train {loss_accum.item():.4f} lr {lr:.2e} tok/s {seq_per_sec:.0f}\n")
                start_time = time.time()
                
            if (step + 1) % args.checkpoint_interval == 0 and rank == 0:
                save_checkpoint(model, optimizer, step, args, rank)
                
                # --- MONITORING TRIGGER ---
                if args.finetune_data_path and monitor:
                    print(f"Step {step}: Triggering fine-tuning monitoring...")
                    finetune_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "finetune.py")
                    step_dir = os.path.join(args.checkpoint_dir, f"step_{step}")
                    history_path = os.path.join(args.log_dir, "finetune_history.csv")
                    
                    try:
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
                    except Exception as e:
                        print(f"Monitoring failed: {e}")
            
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
