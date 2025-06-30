import argparse
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from . import data, models, picme_utils
from . argparser import evaluation_arg_parser
from .data import *

from sklearn.metrics import roc_curve

from captum.attr import IntegratedGradients

# Constants
_DATA_DIR = "<enter your MIMIC IV/CXR data directory here>"
_MODALITY_SHAPES = {"demo": 44, "ts": 76, "projection": 256}
_RENAME_MODALITY = {"mortality": "in-hospital-mortality"}
_TASK_CLASSES = {"mortality": 2, "phenotyping": 25}
_SWEEP_SEEDS = [15, 0, 1, 67, 128, 87, 261, 510, 340, 22]

# Blank strings for all modalities
_PHENO_MODEL_PATHS = {
    ("text_rad", "text_ds"): {
        "fusion_method": "concatenation",
        "state_dict":"",
        "classifier_state_dict": "",
        "embedding_path": "",
    },
    ("text_rad", "text_ds", "demo"): {
        "fusion_method": "concatenation",
        "state_dict": "",
        "classifier_state_dict": "",
        "embedding_path": "",
        "modality_lambdas": [],
    },
    ("text_rad", "text_ds", "img", "demo"): {
        "fusion_method": "concatenation",
        "state_dict": "",
        "classifier_state_dict": "",
        "embedding_path": "",
        "modality_lambdas": [],
    },
    ("text_rad", "text_ds", "img", "ts", "demo"): {
        "fusion_method": "concatenation",
        "state_dict": "",
        "classifier_state_dict": "",
        "embedding_path": "",
        "modality_lambdas": [],
    },
}

_BASELINE_PHENO_MODEL_PATHS = {
    ("text_rad", "text_ds"): {
        "fusion_method": "concatenation",
        "state_dict":"",
        "classifier_state_dict": "",
        "embedding_path": "",
    },
    ("text_ds", "img", "demo"): {
        "fusion_method": "concatenation",
        "state_dict":"",
        "classifier_state_dict": "",
        "embedding_path": "",
        "modality_lambdas": [],
    },
    ("text_rad", "text_ds", "img", "demo"): {
        "fusion_method": "concatenation",
        "state_dict":"",
        "classifier_state_dict": "",
        "embedding_path": "",
        "modality_lambdas": [],
    },
    ("text_rad", "text_ds", "img", "ts", "demo"): {
        "fusion_method": "concatenation",
        "state_dict":"",
        "classifier_state_dict": "",
        "embedding_path": "",
        "modality_lambdas": [],
    },
}

_BASELINE_MLSTM_PHENO = {
    ("text_rad", "text_ds", "img", "ts", "demo"): {
        "state_dict": "",
        "classifier_state_dict": "",
        "fusion_method": "modality_lstm",
        "modality_lambdas": [],
    },
}

_IHM_MODEL_PATHS = {
    ("text_ds", "demo"): {
        "state_dict": "",
        "classifier_state_dict": "",
        "fusion_method": "concatenation",
        "embedding_path": "",
    },

    ("text_ds", "img", "demo"): {
        "state_dict": "",
        "classifier_state_dict": "",
        "fusion_method": "concatenation",
        "embedding_path": "",
        "modality_lambdas": [],
    },
    ("text_rad", "text_ds", "img", "demo"): {
        "state_dict": "",
        "classifier_state_dict": "",
        "fusion_method": "concatenation",
        "embedding_path": "",
        "modality_lambdas": [],
    },
    ("text_rad", "text_ds", "img", "ts", "demo"): {
        "state_dict": "",
        "classifier_state_dict": "",
        "fusion_method": "concatenation",
        "embedding_path": "",
        "modality_lambdas": [],
    },
}


# Blank strings for all modalities
_BASELINE_IHM_PATHS = {
    ("text_ds", "demo"): {
        "state_dict": "",
        "classifier_state_dict": "",
        "fusion_method": "concatenation",
        "embedding_path": "",
        "modality_lambdas": [],
        },
    ("text_ds", "img", "demo"): {
        "state_dict": "",
        "classifier_state_dict": "",
        "fusion_method": "concatenation",
        "embedding_path": "",
        "modality_lambdas": [],
    },
    ("text_rad", "text_ds", "img", "demo"): {
        "state_dict": "",
        "classifier_state_dict": "",
        "fusion_method": "concatenation",
        "embedding_path": "",
        "modality_lambdas": [],
    },
    ("text_rad", "text_ds", "img", "ts", "demo"): {
        "state_dict": "",
        "classifier_state_dict": "",
        "fusion_method": "concatenation",
        "embedding_path": "",
        "modality_lambdas": [],
    },
}

_BASELINE_MLSTM_IHM = {
    ("text_rad", "text_ds", "img", "ts", "demo"): {
        "state_dict": "",
        "classifier_state_dict": "",
        "fusion_method": "modality_lstm",
        "modality_lambdas": [],
    }
}


category_mapping = {
    "Cardiovascular Diseases": [
        "Acute myocardial infarction",
        "Acute cerebrovascular disease",
        "Cardiac dysrhythmias",
        "Conduction disorders",
        "Congestive heart failure; nonhypertensive",
        "Coronary atherosclerosis and other heart disease",
        "Essential hypertension",
        "Hypertension with complications and secondary hypertension",
        "Shock"
    ],
    "Respiratory Diseases": [
        "Chronic obstructive pulmonary disease and bronchiectasis",
        "Other lower respiratory disease",
        "Other upper respiratory disease",
        "Pleurisy; pneumothorax; pulmonary collapse",
        "Pneumonia (except that caused by tuberculosis or sexually transmitted disease)",
        "Respiratory failure; insufficiency; arrest (adult)"
    ],
    "Renal, Metabolic, and Gastrointestinal Diseases": [
        "Acute and unspecified renal failure",
        "Chronic kidney disease",
        "Diabetes mellitus with complications",
        "Diabetes mellitus without complication",
        "Disorders of lipid metabolism",
        "Fluid and electrolyte disorders",
        "Gastrointestinal hemorrhage",
        "Other liver diseases"
    ],
    "Infections, Immune-Related, and Post-Surgical Complications": [
        "Septicemia (except in labor)",
        "Complications of surgical procedures or medical care"
    ]
}

def get_category(phenotype):
    for category, diseases in category_mapping.items():
        if phenotype in diseases:
            return category
    return "Phenotype not found in any category"

# Utility function to build evaluation arguments
def build_eval_arguments(
    model_name,
    modalities,
    fusion_method,
    batch_size,
    metrics,
    task,
    state_dict,
    classifier_state_dict,
    seed_category="single",
    save_prefix="",
    eval_name="",
    modality_lambdas=None,
    seed_number=42
):
    args = [
        "evaluate.py",
        "--model_name", model_name,
        "--modalities", *modalities,
        "--seed_category", seed_category,
        "--fusion_method", fusion_method,
        "--batch_size", str(batch_size),
        "--metrics", *metrics,
        "--task", task,
        "--state_dict", state_dict,
        "--classifier_state_dict", classifier_state_dict,
        "--save_prefix", save_prefix,
        "--eval_name", eval_name
    ]

    if modality_lambdas is not None:
        args.extend(["--modality_lambdas"] + list(map(str, modality_lambdas)))
    if seed_category == "single" and seed_number is not None:
        args.extend(["--seed_number", str(seed_number)])

    sys.argv = args
    return evaluation_arg_parser()

# Function to evaluate a single model
def evaluate_model(args):
    modalities = args.modalities
    task = _RENAME_MODALITY.get(args.task, args.task)
    print(f"Evaluating for task {task}!")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Model selection logic
    if args.model_name == "baseline":
        print("Using fully fine-tuned baseline!")
        model_contrastive = models.MultiModalBaseline(
            ts_input_dim=_MODALITY_SHAPES["ts"],
            demo_input_dim=_MODALITY_SHAPES["demo"],
            projection_dim=256,
            args=args,
            device=device,
        )
        model_contrastive.to(device)
    else:
        # Use the unified MultiModalContrastiveModel for all contrastive approaches
        model_contrastive = models.MultiModalContrastiveModel(
            ts_input_dim=_MODALITY_SHAPES["ts"],
            demo_input_dim=_MODALITY_SHAPES["demo"],
            projection_dim=256,
            ovo=True if args.model_name == "ovo" else False,
            num_modalities=len(modalities),
        )
        model_contrastive.to(device)
        print("Successfully loaded contrastively learned model.")

    # Load state_dict
    if args.model_name == "baseline":
        model_contrastive.load_state_dict(torch.load(args.classifier_state_dict))
        model_contrastive.to(device)
        print("Successfully loaded baseline fully-finetuned model.")
    else:
        model_contrastive.load_state_dict(torch.load(args.state_dict))
        model_contrastive.to(device)
        print("Successfully loaded contrastively learned model.")

    # Load classifier
        classifier = models.ClassificationHead(
            projection_dim=_MODALITY_SHAPES["projection"],
            num_classes=_TASK_CLASSES[args.task],
            fusion_method=args.fusion_method,
            num_modalities=len(args.modalities),
            modality_lambdas=args.modality_lambdas,
        )
        classifier.load_state_dict(torch.load(args.classifier_state_dict))
        classifier.to(device)

    # Set seed and prepare dataloader
    picme_utils.set_seed(args.seed_number)
    _, _, test_input_list = data.get_dataloaders(
        modalities, args.batch_size, _DATA_DIR, task, test=True
    )
    dataloader = data.MergeLoader(test_input_list)

    # Evaluation
    if args.model_name != "baseline":
        classifier.eval()
    model_contrastive.eval()
    all_preds = []
    all_labels = []
    all_outputs = []
    all_embeds = []

    with torch.no_grad():
        for batch in dataloader:
            modalities_data, modalities_type, mask_rad, mask_ds, ts_lens, labels = (
                picme_utils.prep_data(args.modalities, batch, device)
            )

            if args.model_name != "baseline":
                if args.model_name == "single_pair":
                    embeddings1, embeddings2 = model_contrastive(
                        modalities_data[0],
                        modalities_data[1],
                        args.modalities[0],
                        args.modalities[1],
                        mask_rad,
                        mask_ds,
                        ts_lens,
                    )
                    embeddings = [embeddings1, embeddings2]
                else:
                    embeddings = model_contrastive(
                        modalities_data, modalities_type, mask_rad, mask_ds, ts_lens
                    )

                concatenated_embeddings = picme_utils.secure_fusion(
                    embeddings, device, args.fusion_method
                )

                outputs = classifier(concatenated_embeddings)
            else:
                concatenated_embeddings, outputs = model_contrastive.hooked_forward(
                    modalities_data, modalities_type, mask_rad, mask_ds, ts_lens
                )

            preds = picme_utils.predict(outputs, task=args.task)
            all_preds.extend(preds)
            all_outputs.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_embeds.extend(concatenated_embeddings.cpu().numpy())

    return np.array(all_embeds), np.array(all_outputs), np.array(all_preds), np.array(all_labels)

def evaluate_multiple_models(model_paths,
                             batch_size,
                             metrics,
                             task,
                             fusion_method_default=None,
                             baseline=False,
):
    predictions = {}
    all_outputs = {}
    embeddings = {}
    attributions = {}
    ground_truth = None  # This will be set only once to ensure consistency

    for modalities, paths in model_paths.items():        
        # Extract fusion method
        fusion_method = paths.get('fusion_method', fusion_method_default)

        # Dynamically determine the model name
        if len(modalities) == 2:
            model_name = "single_pair"
        elif baseline:
            model_name = "baseline"
        else:
            model_name = "ovo"

        # Build arguments with the determined model name
        print(f"Evaluating modalities: {modalities} using {fusion_method}.")
        args = build_eval_arguments(
            model_name=model_name,
            modalities=modalities,
            fusion_method=fusion_method,
            batch_size=batch_size,
            metrics=metrics,
            task=task,
            state_dict=paths["state_dict"],
            classifier_state_dict=paths["classifier_state_dict"],
            seed_category="single",
            modality_lambdas=paths.get("modality_lambdas"),
        )

        # Evaluate model and get predictions/labels
        embeds, outputs, preds, labels = evaluate_model(args)
        all_outputs[f"{','.join(modalities)}"] = outputs
        embeddings[f"{','.join(modalities)}"] = embeds
        predictions[f"{','.join(modalities)}"] = preds

        # Ensure ground truth is consistent
        if ground_truth is None:
            ground_truth = labels
        else:
            # Validate that the current ground truth matches previous evaluations
            if not np.array_equal(ground_truth, labels):
                raise ValueError(f"Inconsistent ground truth for modalities: {modalities}")

    return embeddings, all_outputs, predictions, ground_truth

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def find_optimal_thresholds_multi_label(y_true, y_pred):
    n_classes = y_true.shape[1]
    optimal_thresholds = []

    for i in range(n_classes):
        fpr, tpr, thresholds = roc_curve(y_true[:, i], y_pred[:, i])
        # Calculate Youden's Index
        youden_index = tpr - fpr
        # Find the threshold that maximizes Youden's Index
        best_idx = np.argmax(youden_index)
        optimal_thresholds.append(thresholds[best_idx])

    return optimal_thresholds

# Function to evaluate models across all seeds
def evaluate_all_seeds(model_paths,
                       batch_size, 
                       metrics,
                       task,
                       sweep_seeds,
                       fusion_method=None,
                       baseline=False,
):
    all_embeddings = {modality: [] for modality in model_paths.keys()}
    all_outputs = {modality: [] for modality in model_paths.keys()}
    all_predictions = {modality: [] for modality in model_paths.keys()}
    
    ground_truth = None  # Ground truth remains consistent across seeds

    for seed in sweep_seeds:
        print(f"Evaluating models for seed: {seed}")
        for modalities, paths in model_paths.items():
            print(f"Evaluating modalities: {modalities}")

            # Update classifier path dynamically based on seed
            if baseline:
                classifier_path = paths["classifier_state_dict"].replace("_0_unfrozen_weighted", f"_{seed}_unfrozen_weighted")
            else:
                classifier_path = paths["classifier_state_dict"].replace("_0_frozen_weighted", f"_{seed}_frozen_weighted")
            print(f"Classifier path: {classifier_path}")
            fusion_method = paths.get("fusion_method", "concatenation")  # Default to "concatenation" if not specified
            print(f"Fusion method: {fusion_method}")

            # Build arguments with the determined model name
            if len(modalities) == 2:
                model_name = "baseline" if baseline else "single_pair"
            else:
                model_name = "baseline" if baseline else "ovo"

            args = build_eval_arguments(
                model_name=model_name,
                modalities=modalities,
                fusion_method=fusion_method,
                batch_size=batch_size,
                metrics=metrics,
                task=task,
                state_dict=paths["state_dict"],
                classifier_state_dict=classifier_path,
                seed_category="single",
                modality_lambdas=paths.get("modality_lambdas"),
                seed_number=seed,
            )

            # Evaluate model and get predictions/labels
            embeds, outputs, preds, labels = evaluate_model(args)
            all_embeddings[modalities].append(embeds)
            all_outputs[modalities].append(outputs)
            
            if task == "phenotyping":
                y_sigmoid = sigmoid(outputs)
                optimal_thresholds = find_optimal_thresholds_multi_label(labels, y_sigmoid)
                y_pred_binary = (y_sigmoid > np.array(optimal_thresholds)).astype(int)
                assert(y_pred_binary.shape == preds.shape)
                all_predictions[modalities].append(y_pred_binary)
            else:
                all_predictions[modalities].append(preds)

            # Ensure ground truth is consistent
            if ground_truth is None:
                ground_truth = labels
            elif not np.array_equal(ground_truth, labels):
                raise ValueError(f"Inconsistent ground truth for modalities: {modalities}")

    return all_predictions, all_embeddings, all_outputs, ground_truth