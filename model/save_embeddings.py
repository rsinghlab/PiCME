from picme_src.model_eval_utils import *
import picme_src.model_eval_utils as eval_utils
import picme_src
import picme_src.data as data
import picme_src.models as models
import picme_src.picme_utils as picme_utils
import picme_src.interp_utils as interp_utils

import torch
import captum
import numpy as np
import pandas as pd
import pickle
import argparse
import sys
import os

# Perform evaluation across all seeds
def get_all_outputs(model_paths, task, baseline=False):
    all_predictions, all_embeddings, all_outputs, ground_truth = evaluate_all_seeds(
        model_paths=model_paths,
        batch_size=32,
        metrics=["auroc", "auprc", "f1"],
        task=task,
        sweep_seeds=eval_utils._SWEEP_SEEDS,
        baseline=baseline,
    )
    return all_predictions, all_embeddings, all_outputs, ground_truth

def save_data(data_dict, data_type, task, modalities, baseline=False, model_type_tag=None, save_dir="picme_src/evals/npz_files"):
    """
    Save data to a pickle file.
    
    Args:
        data_dict: Dictionary containing data to save
        data_type: Type of data to save (embeddings, outputs, predictions)
        task: Task name (e.g., "mortality", "phenotyping")
        modalities: List of modalities
        baseline: Whether the model is a baseline model
        model_type_tag: Explicit model type tag to use in filename (overrides automatic determination)
        save_dir: Base directory to save the data
    """
    # Create specific subdirectory for data type
    specific_dir = f"{save_dir}/{data_type}"
    os.makedirs(specific_dir, exist_ok=True)
    
    # Format task name for file path
    task_name = task
    if task == "mortality":
        task_name = "IHM"
    elif task == "phenotyping":
        task_name = "pheno"
    
    # Create file path
    if model_type_tag:
        model_type = model_type_tag
    else:
        model_type = "baseline" if baseline else "contrastive"
    
    save_pkl = f"{specific_dir}/{task_name}_{','.join(modalities)}_{model_type}_{data_type}.pkl"
    
    # Save data
    with open(save_pkl, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f"Saved {save_pkl} successfully.")

def process_model(task, model_type, save_types):
    """
    Process model and save specified data types.
    
    Args:
        task: Task name (e.g., "mortality", "phenotyping")
        model_type: Model type (e.g., "baseline", "contrastive", "mlstm")
        save_types: List of data types to save (e.g., ["embeddings", "outputs", "predictions"])
    """
    # Determine model paths based on task and model type
    if task == "mortality":
        if model_type == "baseline":
            model_paths = eval_utils._BASELINE_IHM_PATHS
        elif model_type == "mlstm":
            model_paths = eval_utils._BASELINE_MLSTM_IHM
        else:
            model_paths = eval_utils._IHM_MODEL_PATHS
    elif task == "phenotyping":
        if model_type == "baseline":
            model_paths = eval_utils._BASELINE_PHENO_MODEL_PATHS
        elif model_type == "mlstm":
            model_paths = eval_utils._BASELINE_MLSTM_PHENO
        else:
            model_paths = eval_utils._PHENO_MODEL_PATHS
    else:
        raise ValueError(f"Unknown task: {task}")
    
    # Get outputs
    # Treat mlstm as baseline for evaluation but keep the mlstm tag for filenames
    is_baseline = model_type == "baseline" or model_type == "mlstm"
    all_predictions, all_embeddings, all_outputs, ground_truth = get_all_outputs(model_paths, task, baseline=is_baseline)
    
    # Save specified data types
    for modalities in model_paths.keys():
        if "embeddings" in save_types:
            save_data(all_embeddings[modalities], "embeddings", task, modalities, 
                     baseline=is_baseline, model_type_tag=model_type)
        if "outputs" in save_types:
            save_data(all_outputs[modalities], "outputs", task, modalities, 
                     baseline=is_baseline, model_type_tag=model_type)
        if "predictions" in save_types:
            save_data(all_predictions[modalities], "predictions", task, modalities, 
                     baseline=is_baseline, model_type_tag=model_type)
    
    # Save ground truth if requested
    if "ground_truth" in save_types:
        save_dir = "picme_src/evals/npz_files/ground_truth"
        os.makedirs(save_dir, exist_ok=True)
        task_name = "IHM" if task == "mortality" else "pheno"
        save_pkl = f"{save_dir}/{task_name}_ground_truth.pkl"
        with open(save_pkl, 'wb') as f:
            pickle.dump(ground_truth, f)
        print(f"Saved {save_pkl} successfully.")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Save model outputs, embeddings, and predictions.')
    parser.add_argument('--task', type=str, choices=['mortality', 'phenotyping'], required=True,
                        help='Task to evaluate (mortality or phenotyping)')
    parser.add_argument('--model_type', type=str, choices=['baseline', 'contrastive', 'mlstm'], required=True,
                        help='Model type (baseline, contrastive, or mlstm)')
    parser.add_argument('--save_types', type=str, nargs='+', 
                        choices=['embeddings', 'outputs', 'predictions', 'ground_truth'],
                        default=['embeddings'],
                        help='Types of data to save')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    process_model(args.task, args.model_type, args.save_types)