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
# <source your environment>

# Define variables for the arguments
MODEL_NAME="baseline"    # pairs = classic avg, ovo = OvO, single_pair = two mods, baseline=from scratch
MODALITIES=("text_rad" "text_ds" "ts" "demo") #"img" "ts" "demo" "text_rad" "text_ds"
MODALITY_LAMBDAS=() 
LEARNING_RATES=() # 0.000005 0.000001 0.00005 0.00001 0.0005 0.0001
SEED_CATEGORY="single"  # "single" or "sweep"
SEED_NUMBER=42 # only used if SEED_CATEGORY = "single"
FUSION_METHOD="" # choose from: "concatenation", "vanilla_lstm", "modality_lstm"
BATCH_SIZES=(32)
EPOCHS=75
OBJECTIVE="auroc"
METRICS=("auroc" "auprc" "f1") # choose from: auroc, auprc, f1, hamming
WANDB_PROJECT=""
TASK=""  # choose from: "mortality", "phenotyping"
STATE_DICT=""
SAVE_PREFIX=""
FREEZE=false  # FALSE if baseline = true
WEIGH_LOSS=true

# Construct the command with the arguments
COMMAND="python3 finetune.py \
    --model_name $MODEL_NAME \
    --modalities ${MODALITIES[@]} \
    --learning_rate ${LEARNING_RATES[@]} \
    --seed_category $SEED_CATEGORY \
    --fusion_method $FUSION_METHOD \
    --batch_size ${BATCH_SIZES[@]} \
    --epochs $EPOCHS \
    --objective $OBJECTIVE \
    --metrics ${METRICS[@]} \
    --wandb_project $WANDB_PROJECT \
    --task $TASK \
    --state_dict $STATE_DICT \
    --save_prefix $SAVE_PREFIX"

# Add optional arguments if they are true
if [ "$FREEZE" = true ]; then
    COMMAND+=" --freeze"
fi

if [ "$FUSION_METHOD" = "modality_lstm" ]; then
    COMMAND+=" --modality_lambdas ${MODALITY_LAMBDAS[@]}"
fi

if [ "$WEIGH_LOSS" = true ]; then
    COMMAND+=" --weigh_loss"
fi

# Add seed_number if seed_category is single
if [ "$SEED_CATEGORY" = "single" ]; then
    COMMAND+=" --seed_number $SEED_NUMBER"
fi

# Run the command
echo "Running command: $COMMAND"
$COMMAND