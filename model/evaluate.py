import argparse

import numpy as np
import torch
import torch.nn as nn
from picme_src import data, models, picme_utils
from picme_src.argparser import evaluation_arg_parser
from picme_src.data import *

_DATA_DIR = "<your_data_dir>"

_MODALITY_SHAPES = {"demo": 44, "ts": 76, "projection": 256}
_RENAME_MODALITY = {"mortality": "in-hospital-mortality"}

_TASK_CLASSES = {
    "mortality": 2,
    "phenotyping": 25,
}


def path_builder(path, seeds):
    parts = path.split("_")

    pth_type = parts[-1]
    objective_fxn = parts[-2]
    weigh_loss_status = parts[-3]
    freeze_status = parts[-4]
    pre_seed_components = parts[:-5]

    pre_seed_path = "_".join(pre_seed_components)

    all_model_paths = []
    for seed in seeds:
        seed_path = (
            pre_seed_path
            + "_"
            + str(seed)
            + "_"
            + freeze_status
            + "_"
            + weigh_loss_status
            + "_"
            + objective_fxn
            + "_"
            + pth_type
        )

        all_model_paths.append(seed_path)

    return all_model_paths


def evaluate_model(
    dataloader,
    model_contrastive,
    classifier,
    args,
    device,
):
    classifier.eval()
    model_contrastive.eval()

    all_preds = []
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for batch in dataloader:
            modalities_data, modalities_type, mask_rad, mask_ds, ts_lens, labels = (
                picme_utils.prep_data(args.modalities, batch, device)
            )

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
            elif args.model_name != "baseline":
                embeddings = model_contrastive(
                    modalities_data, modalities_type, mask_rad, mask_ds, ts_lens
                )
            if args.model_name != "baseline":
                concatenated_embeddings = picme_utils.secure_fusion(
                    embeddings, device, args.fusion_method
                )
    
                outputs = classifier(concatenated_embeddings)
            else:
                outputs = model_contrastive(modalities_data, modalities_type, mask_rad, mask_ds, ts_lens)

            
            labels = picme_utils.extract_labels(labels, args.task).to(device)

            preds = picme_utils.predict(outputs, task=args.task)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(outputs.cpu().detach().numpy())

    metrics = picme_utils.compute_epoch_metrics(
        args.metrics, args.task, all_labels, all_preds, all_logits, "test"
    )
    print(metrics)
    return metrics


def main(args: argparse.Namespace):
    modalities = args.modalities
    task = (
        args.task if args.task not in _RENAME_MODALITY else _RENAME_MODALITY[args.task]
    )
    print(f"Evaluating for task {task}!")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    if args.model_name == "baseline":
        print("Using fully fine-tuned baseline!")
        model_contrastive = models.MultiModalBaseline(
            ts_input_dim=76,
            demo_input_dim=44,
            projection_dim=256,
            args=args,
            device=device,
        )
        model_contrastive.to(device)
    else:
        model_contrastive = models.MultiModalContrastiveModel(
            ts_input_dim=_MODALITY_SHAPES["ts"],
            demo_input_dim=_MODALITY_SHAPES["demo"],
            projection_dim=256,
            ovo=True if args.model_name == "ovo" else False,
            num_modalities=len(modalities),
        )

    if args.model_name != "baseline":
        model_contrastive.load_state_dict(torch.load(args.state_dict))
        model_contrastive.to(device)
        print("Successfully loaded contrastively learned model.")

    sweep_seeds = [15, 0, 1, 67, 128, 87, 261, 510, 340, 22]
    if args.seed_category == "single":
        seeds = [args.seed_number]
    else:  # args.seed_category == "sweep"
        seeds = sweep_seeds

    model_paths = path_builder(args.classifier_state_dict, seeds)
    metrics_list = {f"test_{metric}": [] for metric in args.metrics}
    metrics_list["test_accuracy"] = []

    for seed, model_path in zip(seeds, model_paths):
        classifier = models.ClassificationHead(
                projection_dim=_MODALITY_SHAPES["projection"],
                num_classes=_TASK_CLASSES[args.task],
                fusion_method=args.fusion_method,
                num_modalities=len(args.modalities),
                modality_lambdas=args.modality_lambdas,
            )

        classifier = classifier.to(device)
        if args.model_name != "baseline":
            classifier.load_state_dict(torch.load(model_path))
            classifier.to(device)
            print(f"Successfully loaded classifier model for seed {seed}.")
        else:
            model_contrastive.load_state_dict(torch.load(model_path))
            model_contrastive.to(device)
            print(f"Successfully loaded classifier model for seed {seed}.")

        print(f"Evaluating with seed: {seed}")
        picme_utils.set_seed(seed)

        _, _, test_input_list = data.get_dataloaders(
            modalities, args.batch_size, _DATA_DIR, task, test=True
        )

        metrics = evaluate_model(
            data.MergeLoader(test_input_list),
            model_contrastive,
            classifier,
            args,
            device,
        )

        for key, value in metrics.items():
            metrics_list[key].append(value)

    if args.seed_category == "sweep":
        metrics_mean_std = {
            metric: (np.mean(values), np.std(values))
            for metric, values in metrics_list.items()
        }
        print("Evaluation Results (Mean ± Std):")
        for metric, (mean, std) in metrics_mean_std.items():
            print(f"{metric}: {mean:.4f} ± {std:.4f}")

        with open(
            f"{args.save_prefix}{args.eval_name}_evaluation_results.txt",
            "w",
        ) as f:
            for metric, (mean, std) in metrics_mean_std.items():
                f.write(f"{metric}: {mean:.4f} ± {std:.4f}\n")
    else:
        print("Evaluation Results for Single Seed:")
        for metric, values in metrics_list.items():
            print(f"{metric}: {values[0]:.4f}")

        with open(
            f"{args.save_prefix}{args.eval_name}_evaluation_results.txt",
            "w",
        ) as f:
            for metric, values in metrics_list.items():
                f.write(f"{metric}: {values[0]:.4f}\n")


if __name__ == "__main__":
    args = evaluation_arg_parser()
    main(args)
