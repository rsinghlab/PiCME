#!/bin/bash

# Request a GPU partition node and access to 1 GPU
#SBATCH -p 3090-gcondo --gres=gpu:1  

# Ensures all allocated cores are on the same node
#SBATCH -N 1

# Request 4 CPU core(s)
#SBATCH -n 8
# Memory is evenly distributed amongst number of cores

#SBATCH --mem=120G
#SBATCH -t 100:00:00

## Provide a job name
#SBATCH -J <job_name>
#SBATCH -o slurm_out/<job_name>_%j.out
#SBATCH -e slurm_out/<job_name>_%j.err

module load python/3.9.16s-x3wdtvt

PRETRAINING_TYPE="" # choose from: single_pair, all_pairs, ovo

if [[ "$PRETRAINING_TYPE" != "single_pair" && "$PRETRAINING_TYPE" != "all_pairs" && "$PRETRAINING_TYPE" != "ovo" ]]; then
    echo "Error: Invalid pretraining type. Must be one of: single_pair, all_pairs, ovo"
    exit 1
fi

# Set parameters
BATCH_SIZE=32
EPOCHS=100
LEARNING_RATE=1e-4
OPTIMIZER="adam"
WEIGHT_DECAY=0.01
TEMPERATURE=0.01
PROJECTION_DIM=256
RANDOM_SEED=42
OUTPUT_DIR="./checkpoints"
WANDB_PROJECT="picme_pretraining"

CMD="python pretrain.py --pretraining_type $PRETRAINING_TYPE \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --optimizer $OPTIMIZER \
    --weight_decay $WEIGHT_DECAY \
    --temperature $TEMPERATURE \
    --projection_dim $PROJECTION_DIM \
    --random_seed $RANDOM_SEED \
    --output_dir $OUTPUT_DIR \
    --wandb_project $WANDB_PROJECT"

if [[ "$PRETRAINING_TYPE" == "single_pair" ]]; then
    if [[ $# -lt 2 ]]; then
        echo "Error: single_pair pretraining requires exactly two modalities"
        exit 1
    fi
    MODALITY1="$1"
    MODALITY2="$2"
    CMD="$CMD --modality_pairs $MODALITY1 $MODALITY2"
    
    echo "Running single pair pretraining with modalities: $MODALITY1 and $MODALITY2"
    
else
    # For all_pairs and ovo, we need at least two modalities
    if [[ $# -lt 2 ]]; then
        echo "Error: all_pairs and ovo pretraining require at least two modalities"
        exit 1
    fi
    
    MODALITIES="$@"
    CMD="$CMD --modalities $MODALITIES"
    
    echo "Running $PRETRAINING_TYPE pretraining with modalities: $MODALITIES"
fi
echo "Executing: $CMD"

eval $CMD
