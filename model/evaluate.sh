#!/bin/bash

# Request a GPU partition node and access to 1 GPU
#SBATCH -p 3090-gcondo --gres=gpu:1  

# Ensures all allocated cores are on the same node
#SBATCH -N 1

# Request 4 CPU core(s)
#SBATCH -n 8
# Memory is evenly distributed amongst number of cores

#SBATCH --mem=80G
#SBATCH -t 5:00:00
#SBATCH -J <job_name>

#SBATCH -o slurm_out/<job_name>_%j.out
#SBATCH -e slurm_out/<job_name>_%j.err

module load python/3.9.16s-x3wdtvt
# <source your environment>

# Define variables for the arguments
MODEL_NAME="ovo"  #  pairs = classic avg, ovo = OvO, pair = two mods, baseline=from scratch
MODALITIES=("text_rad" "text_ds" "img" "ts") # "text_rad" "text_ds" "img" "demo"
MODALITY_LAMBDAS=() 
SEED_CATEGORY="sweep"  # or "sweep"
SEED_NUMBER=42 # only used if SEED_CATEGORY = "single"
FUSION_METHOD="concatenation" # one of "concatenation", "vanilla_lstm", "modality_lstm"
BATCH_SIZE=32
METRICS=("auroc" "auprc" "f1")
TASK="mortality" # phenotyping or mortality

STATE_DICT=""
SAVE_PREFIX=""
EVAL_NAME=""
CLASSIFIER_STATE_DICT=""  # Add path to classifier state dict

# Construct the command with the arguments
COMMAND="python3 evaluate.py \
    --model_name $MODEL_NAME \
    --modalities ${MODALITIES[@]} \
    --seed_category $SEED_CATEGORY \
    --fusion_method $FUSION_METHOD \
    --batch_size $BATCH_SIZE \
    --metrics ${METRICS[@]} \
    --task $TASK \
    --state_dict $STATE_DICT \
    --save_prefix $SAVE_PREFIX \
    --eval_name $EVAL_NAME \
    --classifier_state_dict $CLASSIFIER_STATE_DICT"  # Pass the classifier state dict

# Add optional arguments if they are true
if [ "$FUSION_METHOD" = "modality_lstm" ]; then
    COMMAND+=" --modality_lambdas ${MODALITY_LAMBDAS[@]}"
fi

if [ "$SEED_CATEGORY" = "single" ]; then
    COMMAND+=" --seed_number $SEED_NUMBER"
fi

# Run the command
echo "Running command: $COMMAND"
$COMMAND