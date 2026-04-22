"""Unified DNABERT2 training entrypoint.

Quick usage:
    # BERT architecture
    torchrun --nproc_per_node=4 main_train.py --architecture bert --batch-size 64 --total-batch-size 4096

    # MAELM architecture
    torchrun --nproc_per_node=1 main_train.py --architecture maelm --batch-size 64 --total-batch-size 4096
"""

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

import argparse
import datetime
import glob
import inspect
import io
import math
import os
import time
import warnings
from dataclasses import dataclass
from typing import Optional

import monitor
import numpy as np
import torch
import torch.distributed as dist
import webdataset as wds
from torch import nn

try:
    import wandb
except ImportError:
    wandb = None


@dataclass
class TrainingConfig:
    architecture: str = "bert"

    total_batch_size: int = 4096
    batch_size: int = 8
    max_seq_length: int = 512
    alibi_starting_size: int = 512

    max_lr: float = 5e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 600
    max_steps: int = 4000
    weight_decay: float = 0.1
    mask_ratio: float = 0.15

    checkpoint_interval: int = 500
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "log"
    train_shards_pattern: str = "shards/train-*.tar"

    use_compile: bool = True
    shuffle_buffer: int = 10000

    num_special_tokens: int = 5
    mask_token_id: int = 4
    vocab_size: int = 4096

    finetune_data_path: Optional[str] = None
    num_workers: int = 4

    maelm_model_type: str = "maelm_v2"
    maelm_encoder_layers: int = 12
    maelm_decoder_layers: int = 12
    scheduler: str = "auto"

    use_wandb: bool = True
    wandb_project: str = "dnabert2-training"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_mode: str = "online"
    wandb_existing_run_path: Optional[str] = None


def get_num_cpu_available():
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        warnings.warn(
            "Unable to determine number of available CPUs for this process. Falling back to os.cpu_count().",
            RuntimeWarning,
            stacklevel=2,
        )
        return os.cpu_count()


def npy_decoder(data: bytes) -> torch.Tensor:
    return torch.from_numpy(np.load(io.BytesIO(data), allow_pickle=False))


def make_dataset(pattern: str, shuffle_buf: int = 10_000, resampled: bool = False, world_size: int = 1):
    if world_size > 1:
        ds = wds.WebDataset(
            pattern,
            resampled=resampled,
            nodesplitter=wds.shardlists.split_by_node,
        )
    else:
        ds = wds.WebDataset(pattern, resampled=resampled, shardshuffle=True)

    if resampled:
        ds = ds.with_epoch(1000000)
        ds = ds.with_length(1000000)

    ds = ds.shuffle(shuffle_buf)
    ds = ds.to_tuple("__key__", "tokens", "attention_mask").map_tuple(
        lambda x: x,
        npy_decoder,
        npy_decoder,
    )
    return ds


def setup_distributed():
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "CUDA is required for distributed training"
        init_process_group(backend="nccl", timeout=datetime.timedelta(hours=3))
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        device_type = "cuda"
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True

        device = "cpu"
        use_cuda = False
        if torch.cuda.is_available():
            device = "cuda"
            use_cuda = True
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        print(f"using device: {device}")

        device_type = "cuda" if device.startswith("cuda") else "cpu"

        torch.manual_seed(123)
        if use_cuda:
            torch.cuda.manual_seed(123)

    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, device_type, master_process


def setup_data_loader(config: TrainingConfig, ddp_world_size: int, device: str, master_process: bool):
    assert config.total_batch_size % (config.batch_size * ddp_world_size) == 0
    grad_accum_steps = config.total_batch_size // (config.batch_size * ddp_world_size)

    if master_process:
        print(f"total desired batch size: {config.total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    train_dataset_pattern = sorted(glob.glob(config.train_shards_pattern))
    if not train_dataset_pattern:
        train_dataset_pattern = config.train_shards_pattern

    train_dataset = make_dataset(
        train_dataset_pattern,
        config.shuffle_buffer,
        resampled=True,
        world_size=ddp_world_size,
    )

    cpu_workers = min(get_num_cpu_available(), config.num_workers)

    if ddp_world_size > 1:
        dataloader_train = iter(
            wds.WebLoader(
                train_dataset.batched(config.batch_size, partial=False),
                num_workers=cpu_workers,
                batch_size=None,
            )
        )
    else:
        dl_train_kwargs = {
            "batch_size": config.batch_size,
            "drop_last": True,
            "sampler": None,
            "shuffle": False,
            "num_workers": cpu_workers,
            "pin_memory": device != "cpu",
        }
        dataloader_train = iter(torch.utils.data.DataLoader(train_dataset, **dl_train_kwargs))

    return dataloader_train, grad_accum_steps


def process_batch_mlm(batch, config: TrainingConfig):
    _, input_ids, att_mask = batch

    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.as_tensor(input_ids)
    if not isinstance(att_mask, torch.Tensor):
        att_mask = torch.as_tensor(att_mask)

    targets = input_ids.long()
    att_mask = att_mask.to(dtype=torch.float32)

    if targets.size(1) > config.max_seq_length:
        targets = targets[:, : config.max_seq_length]
        att_mask = att_mask[:, : config.max_seq_length]

    if (targets >= config.vocab_size).any():
        targets = torch.clamp(targets, max=config.vocab_size - 1)

    valid_token_mask = targets > config.num_special_tokens
    random_mask = torch.rand(targets.shape)
    input_maskout = random_mask < config.mask_ratio
    input_maskout &= valid_token_mask

    masked_input = targets.detach().clone()
    masked_input[input_maskout] = config.mask_token_id
    att_mask[input_maskout] = 0.0

    return masked_input, att_mask, targets


def create_model(config: TrainingConfig, device: str, ddp: bool, ddp_local_rank: int):
    from transformers import BertConfig

    torch.set_float32_matmul_precision("high")

    encoder_config = BertConfig(
        **{
            "_name_or_path": "zhihan1996/DNABERT-2-117M",
            "alibi_starting_size": config.alibi_starting_size,
            "attention_probs_dropout_prob": 0.0,
            "classifier_dropout": None,
            "gradient_checkpointing": False,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": config.max_seq_length,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "position_embedding_type": "absolute",
            "torch_dtype": "float32",
            "transformers_version": "4.29.0",
            "type_vocab_size": 2,
            "use_cache": True,
            "vocab_size": config.vocab_size,
            "num_labels": config.vocab_size,
        }
    )

    decoder_config = BertConfig(
        **{
            "_name_or_path": "zhihan1996/DNABERT-2-117M",
            "alibi_starting_size": config.alibi_starting_size,
            "attention_probs_dropout_prob": 0.0,
            "classifier_dropout": None,
            "gradient_checkpointing": False,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": config.max_seq_length,
            "num_attention_heads": 12,
            "num_hidden_layers": config.maelm_decoder_layers,
            "position_embedding_type": "absolute",
            "torch_dtype": "float32",
            "transformers_version": "4.29.0",
            "type_vocab_size": 2,
            "use_cache": True,
            "vocab_size": config.vocab_size,
            "num_labels": config.vocab_size,
        }
    )

    encoder_config.num_hidden_layers = config.maelm_encoder_layers

    if config.architecture == "bert":
        from bert_layers import BertForMaskedLM

        model = BertForMaskedLM(encoder_config)
    elif config.architecture == "maelm":
        from maelm_model import MAELMModel

        model = MAELMModel(encoder_config, decoder_config)
    else:
        raise ValueError(f"Unknown architecture: {config.architecture}")

    model.to(device)
    if config.use_compile:
        model = torch.compile(model)

    if ddp:
        find_unused = config.architecture == "maelm"
        model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=find_unused)

    raw_model = model.module if ddp else model
    return model, raw_model, encoder_config


def get_lr(step: int, config: TrainingConfig):
    min_lr = config.max_lr * config.min_lr_ratio

    if step < config.warmup_steps:
        return config.max_lr * (step + 1) / config.warmup_steps
    if step > config.max_steps:
        return min_lr
    return config.max_lr + ((min_lr - config.max_lr) / (config.max_steps - config.warmup_steps)) * (
        step - config.warmup_steps
    )


def configure_optimizers(model, config: TrainingConfig, device_type: str, master_process: bool):
    if config.architecture == "maelm":
        if master_process:
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"maelm optimizer on {trainable_params:,} trainable parameters")
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.max_lr,
            weight_decay=config.weight_decay,
        )

    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

    decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    if master_process:
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )

    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    if master_process:
        print(f"using fused AdamW: {use_fused}")

    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=config.max_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=use_fused,
    )
    return optimizer


def build_scheduler(optimizer, config: TrainingConfig, start_step: int):
    if config.scheduler == "linear":
        return None

    use_cosine = config.scheduler == "cosine" or (config.scheduler == "auto" and config.architecture == "maelm")
    if not use_cosine:
        return None

    from transformers import get_cosine_schedule_with_warmup

    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_steps,
        last_epoch=start_step - 1,
    )


def setup_checkpoint_dirs(config: TrainingConfig):
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    checkpoint_path = os.path.join(config.checkpoint_dir, "latest_checkpoint.pt")
    log_file = os.path.join(config.log_dir, "log.txt")
    return checkpoint_path, log_file


def save_checkpoint(step: int, raw_model, optimizer, loss: float, dnabert_config, checkpoint_path: str, config: TrainingConfig, master_process: bool):
    if not master_process:
        return

    checkpoint = {
        "step": step,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "config": dnabert_config.to_dict(),
        "architecture": config.architecture,
        "maelm_model_type": config.maelm_model_type,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at step {step}")

    if config.architecture == "maelm":
        step_dir = os.path.join(config.checkpoint_dir, f"step_{step}")
        os.makedirs(step_dir, exist_ok=True)

        torch.save(raw_model.state_dict(), os.path.join(step_dir, "pytorch_model.bin"))
        dnabert_config.save_pretrained(step_dir)
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "step": step,
                "architecture": "maelm",
            },
            os.path.join(step_dir, "training_state.pt"),
        )

        latest_link = os.path.join(config.checkpoint_dir, "latest")
        if os.path.islink(latest_link) or os.path.exists(latest_link):
            try:
                os.remove(latest_link)
            except OSError:
                pass
        os.symlink(os.path.abspath(step_dir), latest_link)


def load_checkpoint(raw_model, optimizer, checkpoint_path: str, checkpoint_dir: str, device: str, master_process: bool):
    if os.path.exists(checkpoint_path):
        if master_process:
            print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        raw_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = checkpoint["step"] + 1
        if master_process:
            print(f"Resumed from step {start_step}, previous loss: {checkpoint['loss']:.6f}")
        return start_step

    legacy_latest = os.path.join(checkpoint_dir, "latest")
    legacy_train_state = os.path.join(legacy_latest, "training_state.pt")
    legacy_model_state = os.path.join(legacy_latest, "pytorch_model.bin")
    if os.path.islink(legacy_latest) and os.path.exists(legacy_train_state) and os.path.exists(legacy_model_state):
        if master_process:
            print(f"Loading legacy MAELM checkpoint from {legacy_latest}")
        train_state = torch.load(legacy_train_state, map_location=device)
        model_state = torch.load(legacy_model_state, map_location=device)
        raw_model.load_state_dict(model_state)
        optimizer.load_state_dict(train_state["optimizer"])
        start_step = int(train_state["step"]) + 1
        if master_process:
            print(f"Resumed from legacy checkpoint at step {start_step}")
        return start_step

    if master_process:
        print("No checkpoint found, starting from step 0")
    return 0


def maybe_init_wandb(config: TrainingConfig, master_process: bool):
    if not (config.use_wandb and master_process):
        return None

    if wandb is None:
        print("wandb is not installed. Install it or pass --no-wandb to suppress this warning.")
        return None

    run_config = {
        "architecture": config.architecture,
        "total_batch_size": config.total_batch_size,
        "batch_size": config.batch_size,
        "max_seq_length": config.max_seq_length,
        "max_lr": config.max_lr,
        "min_lr_ratio": config.min_lr_ratio,
        "warmup_steps": config.warmup_steps,
        "max_steps": config.max_steps,
        "weight_decay": config.weight_decay,
        "mask_ratio": config.mask_ratio,
        "checkpoint_interval": config.checkpoint_interval,
        "train_shards_pattern": config.train_shards_pattern,
        "shuffle_buffer": config.shuffle_buffer,
        "maelm_model_type": config.maelm_model_type,
        "maelm_encoder_layers": config.maelm_encoder_layers,
        "maelm_decoder_layers": config.maelm_decoder_layers,
    }

    init_kwargs = {
        "project": config.wandb_project,
        "entity": config.wandb_entity,
        "name": config.wandb_run_name,
        "mode": config.wandb_mode,
        "config": run_config,
    }

    if config.wandb_existing_run_path:
        api = wandb.Api()
        existing_run = api.run(config.wandb_existing_run_path)
        for key, value in run_config.items():
            existing_run.config[key] = value
        existing_run.update()

        init_kwargs["id"] = existing_run.id
        init_kwargs["resume"] = "allow"

        path_parts = config.wandb_existing_run_path.split("/")
        if len(path_parts) >= 3:
            init_kwargs["entity"] = path_parts[0]
            init_kwargs["project"] = path_parts[1]

    run = wandb.init(**init_kwargs)
    return run


def maybe_log_wandb(metrics: dict, step: int, run):
    if run is not None:
        run.log(metrics, step=step)


def train(config: TrainingConfig):
    ddp, _, ddp_local_rank, ddp_world_size, device, device_type, master_process = setup_distributed()

    dataloader_train, grad_accum_steps = setup_data_loader(config, ddp_world_size, device, master_process)

    model, raw_model, dnabert_config = create_model(config, device, ddp, ddp_local_rank)

    optimizer = configure_optimizers(raw_model, config, device_type, master_process)

    checkpoint_path, log_file = setup_checkpoint_dirs(config)
    start_step = load_checkpoint(raw_model, optimizer, checkpoint_path, config.checkpoint_dir, device, master_process)
    scheduler = build_scheduler(optimizer, config, start_step)

    log_mode = "a" if start_step > 0 else "w"
    with open(log_file, log_mode, encoding="utf-8"):
        pass

    criterion = nn.CrossEntropyLoss() if config.architecture == "bert" else None

    wandb_run = maybe_init_wandb(config, master_process)

    if master_process:
        print(f"Starting {config.architecture} training from step {start_step} to {config.max_steps}")

    for step in range(start_step, config.max_steps):
        t0 = time.time()
        last_step = step == config.max_steps - 1

        model.train()
        optimizer.zero_grad()
        loss_accum = torch.tensor(0.0, device=device)

        for micro_step in range(grad_accum_steps):
            try:
                batch = next(dataloader_train)
                input_ids, attention_mask, targets = process_batch_mlm(batch, config)
            except (StopIteration, Exception):
                train_dataset_pattern = sorted(glob.glob(config.train_shards_pattern))
                if not train_dataset_pattern:
                    train_dataset_pattern = config.train_shards_pattern
                train_dataset = make_dataset(
                    train_dataset_pattern,
                    config.shuffle_buffer,
                    resampled=True,
                    world_size=ddp_world_size,
                )

                cpu_workers = min(get_num_cpu_available(), config.num_workers)
                if ddp_world_size > 1:
                    dataloader_train = iter(
                        wds.WebLoader(
                            train_dataset.batched(config.batch_size, partial=False),
                            num_workers=cpu_workers,
                            batch_size=None,
                        )
                    )
                else:
                    dl_kwargs = {
                        "batch_size": config.batch_size,
                        "drop_last": True,
                        "sampler": None,
                        "shuffle": False,
                        "num_workers": cpu_workers,
                        "pin_memory": device != "cpu",
                    }
                    dataloader_train = iter(torch.utils.data.DataLoader(train_dataset, **dl_kwargs))

                batch = next(dataloader_train)
                input_ids, attention_mask, targets = process_batch_mlm(batch, config)

            input_ids = input_ids.to(device=device)
            attention_mask = attention_mask.to(dtype=torch.float32, device=device)
            targets = targets.to(device=device)

            if ddp:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                if config.architecture == "bert":
                    output = model(input_ids, attention_mask=attention_mask)
                    masked_tokens_idx = input_ids == config.mask_token_id
                    if masked_tokens_idx.any():
                        loss = criterion(
                            output.logits.view(-1, config.vocab_size)[masked_tokens_idx.view(-1)],
                            targets.view(-1)[masked_tokens_idx.view(-1)],
                        )
                    else:
                        loss = output.logits.new_tensor(0.0)
                else:
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=targets,
                        model_type=config.maelm_model_type,
                    )
                    if hasattr(output, "loss"):
                        loss = output.loss
                    elif isinstance(output, (tuple, list)):
                        loss = output[0]
                    else:
                        loss = output

            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if scheduler is None:
            lr = get_lr(step, config)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]

        if device_type == "cuda":
            torch.cuda.synchronize()

        dt = time.time() - t0
        sequences_processed = input_ids.shape[0] * grad_accum_steps * ddp_world_size
        sequences_per_sec = sequences_processed / max(dt, 1e-6)

        if master_process:
            loss_value = loss_accum.item()
            norm_value = norm.item() if isinstance(norm, torch.Tensor) else float(norm)
            print(
                f"step {step:5d} | loss: {loss_value:.6f} | lr {lr:.4e} | norm: {norm_value:.4f} | "
                f"dt: {dt*1000:.2f}ms | seq/sec: {sequences_per_sec:.2f}"
            )
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{step} train {loss_value:.6f}\n")

            maybe_log_wandb(
                {
                    "train/loss": loss_value,
                    "train/lr": lr,
                    "train/grad_norm": norm_value,
                    "train/sequences_per_sec": sequences_per_sec,
                },
                step,
                wandb_run,
            )

        if (step + 1) % config.checkpoint_interval == 0 or last_step:
            save_checkpoint(
                step,
                raw_model,
                optimizer,
                loss_accum.item(),
                dnabert_config,
                checkpoint_path,
                config,
                master_process,
            )

            if master_process and config.finetune_data_path:
                print(f"Triggering fine-tuning evaluation at step {step}...")
                history_path = os.path.join(config.log_dir, "finetune_history.csv")
                finetune_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "finetune.py")

                model_type = "maelm" if config.architecture == "maelm" else "bert"
                metrics = monitor.run_finetune(
                    finetune_script,
                    checkpoint_path,
                    config.finetune_data_path,
                    config.log_dir,
                    step,
                    model_type=model_type,
                )

                if metrics:
                    print(f"Fine-tuning step {step} completed with metrics: {metrics}")
                    monitor.update_finetune_history(history_path, step, metrics)
                    monitor.plot_curves(log_file, history_path, config.log_dir)
                    maybe_log_wandb({f"finetune/{k}": v for k, v in metrics.items()}, step, wandb_run)
                else:
                    print(f"Fine-tuning step {step} failed or returned no metrics.")

            if ddp:
                dist.barrier()

    if wandb_run is not None:
        wandb_run.finish()

    if ddp:
        destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified trainer for DNABERT2 (BERT or MAELM)",
        epilog=(
            "Examples:\n"
            "  torchrun --nproc_per_node=4 main_train.py --architecture bert --batch-size 64 --total-batch-size 4096\n"
            "  torchrun --nproc_per_node=1 main_train.py --architecture maelm --batch-size 64 --total-batch-size 4096"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", -1)))

    parser.add_argument("--architecture", type=str, choices=["bert", "maelm"], default="bert")

    parser.add_argument("--total-batch-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--alibi-starting-size", type=int, default=512)

    parser.add_argument("--max-lr", type=float, default=5e-4)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=30000)
    parser.add_argument("--max-steps", type=int, default=500000)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--mask-ratio", type=float, default=0.15)

    parser.add_argument("--checkpoint-interval", type=int, default=2000)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="log")

    parser.add_argument("--train-shards-pattern", type=str, default="shards/train-*.tar")

    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--shuffle-buffer", type=int, default=10000)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--maelm-model-type", type=str, default="maelm_v2", choices=["maelm_v1", "maelm_v2", "baseline"])
    parser.add_argument("--maelm-encoder-layers", type=int, default=12)
    parser.add_argument("--maelm-decoder-layers", type=int, default=12)
    parser.add_argument("--scheduler", type=str, default="auto", choices=["auto", "linear", "cosine"])

    parser.add_argument("--finetune-data-path", type=str, default=None)

    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="dnabert2-training")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--wandb-existing-run-path", type=str, default=None)

    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Ignoring unknown arguments: {unknown}")
    if args.learning_rate is not None:
        args.max_lr = args.learning_rate
    return args


def main():
    args = parse_args()

    config = TrainingConfig(
        architecture=args.architecture,
        total_batch_size=args.total_batch_size,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        alibi_starting_size=args.alibi_starting_size,
        max_lr=args.max_lr,
        min_lr_ratio=args.min_lr_ratio,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        weight_decay=args.weight_decay,
        mask_ratio=args.mask_ratio,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        train_shards_pattern=args.train_shards_pattern,
        use_compile=not args.no_compile,
        shuffle_buffer=args.shuffle_buffer,
        finetune_data_path=args.finetune_data_path,
        num_workers=args.num_workers,
        maelm_model_type=args.maelm_model_type,
        maelm_encoder_layers=args.maelm_encoder_layers,
        maelm_decoder_layers=args.maelm_decoder_layers,
        scheduler=args.scheduler,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_mode=args.wandb_mode,
        wandb_existing_run_path=args.wandb_existing_run_path,
    )

    train(config)


if __name__ == "__main__":
    main()
