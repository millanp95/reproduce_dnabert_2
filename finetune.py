import os
import csv
import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Sequence, Tuple, List, Union

import torch
import transformers
import sklearn
import numpy as np
from torch.utils.data import Dataset
from bert_layers import BertForSequenceClassification

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    TaskType,
)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    model_type: str = field(default="bert", metadata={"help": "Model type: 'bert' or 'maelm'"})
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"})
    use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    lora_target_modules: str = field(default="query,value", metadata={"help": "where to perform LoRA"})


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    kmer: int = field(default=-1, metadata={"help": "k-mer for input sequence. -1 means not using k-mer."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    num_train_epochs: int = field(default=1)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=100)
    save_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    evaluation_strategy: str = field(default="steps"),
    warmup_steps: int = field(default=50)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=1e-4)
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=True)
    output_dir: str = field(default="output")
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=True)
    seed: int = field(default=42)
    evaluation_strategy: str = "no",
    save_strategy: str = 'no' 

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


"""
Get the reversed complement of the original DNA sequence.
"""
def get_alter_of_dna_sequence(sequence: str):
    MAP = {"A": "T", "T": "A", "C": "G", "G": "C"}
    # return "".join([MAP[c] for c in reversed(sequence)])
    return "".join([MAP[c] for c in sequence])

"""
Transform a dna sequence to k-mer string
"""
def generate_kmer_str(sequence: str, k: int) -> str:
    """Generate k-mer string from DNA sequence."""
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])


"""
Load or generate k-mer string for each DNA sequence. The generated k-mer string will be saved to the same directory as the original data with the same name but with a suffix of "_{k}mer".
"""
def load_or_generate_kmer(data_path: str, texts: List[str], k: int) -> List[str]:
    """Load or generate k-mer string for each DNA sequence."""
    kmer_path = data_path.replace(".csv", f"_{k}mer.json")
    if os.path.exists(kmer_path):
        logging.warning(f"Loading k-mer from {kmer_path}...")
        with open(kmer_path, "r") as f:
            kmer = json.load(f)
    else:    
        logging.warning(f"Generating k-mer...")
        kmer = [generate_kmer_str(text, k) for text in texts]
        with open(kmer_path, "w") as f:
            json.dump(kmer, f)
    return kmer

class SupervisedDataset(Dataset):
    """Dataset for supervised learning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, kmer: int = -1):
        super(SupervisedDataset, self).__init__()

        # load data from the disk
        with open(data_path, "r") as f:
            data = list(csv.reader(f, delimiter="\t"))[1:]
        if len(data[0]) == 2:
            # data is in the format of [text, label]
            logging.warning("Perform single sequence classification...")
            texts = [d[0] for d in data]
            labels = [int(d[1]) for d in data]
        elif len(data[0]) == 3:
            # data is in the format of [text1, text2, label]
            logging.warning("Perform sequence-pair classification...")
            texts = [[d[0], d[1]] for d in data]
            labels = [int(d[2]) for d in data]
        else:
            raise ValueError("Data format not supported.")
        
        if kmer != -1:
            # only write file on the first process
            if torch.distributed.get_rank() == 0:
                load_or_generate_kmer(data_path, texts, kmer)
            torch.distributed.barrier()
            texts = load_or_generate_kmer(data_path, texts, kmer)

        output = tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
        )

        self.input_ids = output["input_ids"]
        self.attention_mask = output["attention_mask"]
        self.labels = labels
        self.num_labels = len(set(labels))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i], attention_mask=self.attention_mask[i])

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # load tokenizer
    if model_args.tokenizer_name:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.tokenizer_name,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )

    # load data
    dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, kmer=data_args.kmer)
    
    # load model
    num_labels = dataset.num_labels
    print(f"Number of labels: {num_labels}")

    if model_args.model_type == "maelm":
        print("Loading MAELM checkpoint for classification...")
        # MAELM checkpoints have 'encoder.' prefix for the BERT part
        # We need to load it into BertForSequenceClassification
        # 1. Load config
        if os.path.exists(model_args.model_name_or_path):
            config_path = model_args.model_name_or_path
        else:
             config_path = "zhihan1996/DNABERT-2-117M" # Fallback if just loading from weights file without config.json

        config = transformers.AutoConfig.from_pretrained(
            config_path,
            num_labels=num_labels,
            finetuning_task=data_args.data_path,
            cache_dir=training_args.cache_dir,
            trust_remote_code=True,
        )
        
        # 2. Initialize standard classification model
        model = BertForSequenceClassification(config)
        
        # 3. Load weights manually handling prefix mismatch
        # We expect model_name_or_path to be a DIRECTORY containing pytorch_model.bin
        # OR it can be the .bin file itself
        if os.path.isdir(model_args.model_name_or_path):
            ckpt_file = os.path.join(model_args.model_name_or_path, "pytorch_model.bin")
        else:
            ckpt_file = model_args.model_name_or_path
            
        print(f"Loading weights from {ckpt_file}")
        state_dict = torch.load(ckpt_file, map_location="cpu")
        if "model" in state_dict: state_dict = state_dict["model"] # Handle wrapper if present
        
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("encoder."):
                new_key = k.replace("encoder.", "bert.")
                new_state_dict[new_key] = v
            elif k.startswith("classifier."): # If resuming finetuning
                new_state_dict[k] = v
            # Ignore decoder.
                
        # Load
        if new_state_dict:
            missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
            print(f"MAELM Load | Missing: {len(missing)} | Unexpected: {len(unexpected)}")
        else:
            print("Warning: No 'encoder.' keys found in checkpoint. Trying direct load...")
            # Maybe it was already converted or saving was different
            model.load_state_dict(state_dict, strict=False)
            
    else:
        # Standard loading
        model = BertForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            num_labels=num_labels,
            cache_dir=training_args.cache_dir,
            trust_remote_code=True,
        )
    
    # Configure LoRA if requested
    if model_args.use_lora:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=model_args.lora_target_modules.split(","),
            bias="none",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # define trainer
    trainer = transformers.Trainer(model=model, tokenizer=tokenizer, args=training_args, train_dataset=dataset)
    trainer.train()

    # save model
    if training_args.save_model:
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    # evaluate
    if training_args.eval_and_save_results:
        results = trainer.evaluate()
        with open(os.path.join(training_args.output_dir, "eval_results.json"), "w") as f:
            json.dump(results, f)

if __name__ == "__main__":
    train()
