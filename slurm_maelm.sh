#!/bin/bash
#SBATCH --account=def-lila-ab
#SBATCH --job-name=maelm_baseline
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mem=64G
#SBATCH --time=24:55:00
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# --- Environment ---

export CUDA_VISIBLE_DEVICES=0
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export PYTHONUNBUFFERED=1

export HF_HOME=/scratch/m4safari/cache/huggingface
export TRANSFORMERS_CACHE=/scratch/m4safari/cache/huggingface
export MPLCONFIGDIR=/scratch/m4safari/cache/matplotlib
mkdir -p "$HF_HOME" "$MPLCONFIGDIR"

module load cuda
module load cudnn
module load python/3.11
module load scipy-stack
module load arrow

source /scratch/m4safari/BarcodeMAE_venv/bin/activate

if [ -z "$SLURM_CPUS_PER_TASK" ]; then
    export OMP_NUM_THREADS=8
else
    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi

# --- Paths ---

SHARDS_DIR="/scratch/m4safari/dnabert2_wds/shards_0.01"
CHECKPOINT_DIR="/scratch/m4safari/checkpoints/maelm_baseline"
LOG_DIR="/scratch/m4safari/logs/maelm_baseline"

mkdir -p "$CHECKPOINT_DIR" "$LOG_DIR"

export MASTER_ADDR=$(hostname)
export MASTER_PORT=12346
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

echo "------------------------------------------------------"
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $(hostname)"
echo "Date:     $(date)"
echo "Shards:   $SHARDS_DIR"
echo "------------------------------------------------------"

# --- Training ---
# Architecture  : MAELM encoder-decoder, no taxonomy head
# Encoder/Decoder : 6 layers, 6 attention heads
# Compile       : enabled

torchrun --nproc_per_node=1 \
         --nnodes=1 \
         --node_rank=0 \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         main_train.py \
         --architecture maelm \
         \
         --n-layers 6 \
         --n-heads 6 \
         --decoder-n-layers 6 \
         --decoder-n-heads 6 \
         \
         --batch-size 128 \
         --total-batch-size 4096 \
         --max-steps 10 \
         --warmup-steps 30000 \
         --max-lr 5e-4 \
         --weight-decay 0.1 \
         --mask-ratio 0.15 \
         --max-seq-length 224 \
         \
         --checkpoint-interval 2000 \
         --log-interval 1 \
         --compile-mode reduce-overhead \
         --num-workers 4 \
         --train-shards-pattern "$SHARDS_DIR/train-*.tar" \
         --checkpoint-dir "$CHECKPOINT_DIR" \
         --log-dir "$LOG_DIR"