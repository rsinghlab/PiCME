#!/bin/bash

# Request a GPU partition node and access to 1 GPU
#SBATCH -p 3090-gcondo --gres=gpu:1 

# Ensures all allocated cores are on the same node
#SBATCH -N 1

# Request 4 CPU core(s)
#SBATCH -n 4

#SBATCH --mem=40G
#SBATCH -t 2:00:00

## Provide a job name
#SBATCH -J save_model_data
#SBATCH -o slurm_out/save_model_data_%j.out
#SBATCH -e slurm_out/save_model_data_%j.err

module load python/3.9.16s-x3wdtvt
# source your environment.

# Default values
TASK="mortality"
MODEL_TYPE="baseline"
SAVE_TYPES="embeddings"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --task)
      TASK="$2"
      shift 2
      ;;
    --model_type)
      MODEL_TYPE="$2"
      shift 2
      ;;
    --save_types)
      SAVE_TYPES="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--task mortality|phenotyping] [--model_type baseline|contrastive|mlstm] [--save_types 'embeddings outputs predictions ground_truth']"
      exit 1
      ;;
  esac
done

# Validate task
if [[ "$TASK" != "mortality" && "$TASK" != "phenotyping" ]]; then
  echo "Invalid task: $TASK. Must be 'mortality' or 'phenotyping'"
  exit 1
fi

# Validate model type
if [[ "$MODEL_TYPE" != "baseline" && "$MODEL_TYPE" != "contrastive" && "$MODEL_TYPE" != "mlstm" ]]; then
  echo "Invalid model type: $MODEL_TYPE. Must be 'baseline', 'contrastive', or 'mlstm'"
  exit 1
fi

# Update job name to reflect parameters
job_name="save_${TASK}_${MODEL_TYPE}_${SAVE_TYPES// /_}"
scontrol update job $SLURM_JOB_ID name=$job_name

echo "Running with parameters:"
echo "Task: $TASK"
echo "Model Type: $MODEL_TYPE"
echo "Save Types: $SAVE_TYPES"

# Run the Python script with the specified parameters
python3 save_embeddings.py --task "$TASK" --model_type "$MODEL_TYPE" --save_types $SAVE_TYPES