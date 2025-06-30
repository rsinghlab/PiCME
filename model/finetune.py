import argparse
import logging
import pprint
import time

import torch
import torch.nn as nn
import wandb
from picme_src import data, models, picme_utils
from picme_src.argparser import finetune_arg_parser
from picme_src.data import *
import gc
torch.cuda.empty_cache()
gc.collect()

_DATA_DIR = "<your_data_dir>"

_MODALITY_SHAPES = {"demo": 44, "ts": 76, "projection": 256}
_RENAME_MODALITY = {"mortality": "in-hospital-mortality"}

_TASK_WEIGHTS = {
    "in-hospital-mortality": data._MORTALITY_CLASS_WEIGHTS,
    "phenotyping": data._PHENO_CLASS_WEIGHTS,
}

_TASK_CLASSES = {
    "mortality": 2,
    "phenotyping": 25,
}


def train_model(
    dataloaders_dict,
    criterion,
    model_contrastive,
    config,
    path,
    args,
    device,
):
    torch.cuda.empty_cache()
    picme_utils.set_seed(config.random_seed)

    projection_dim = _MODALITY_SHAPES["projection"]

    classifier = models.ClassificationHead(
        projection_dim=projection_dim,
        num_classes=_TASK_CLASSES[args.task],
        fusion_method=args.fusion_method,
        num_modalities=len(args.modalities),
        modality_lambdas=args.modality_lambdas,
    )
    classifier = classifier.to(device)

    num_epochs = config.epochs
    print(config.optimizer, config.learning_rate)
    if args.model_name != "baseline":
        optimizer = picme_utils.build_optimizer(
            classifier, config.optimizer, config.learning_rate, momentum=0.01
        )
    else:
        optimizer = picme_utils.build_optimizer(
            model_contrastive, config.optimizer, config.learning_rate, momentum=0.01
        )
    since = time.time()

    best_metric = 0.0
    patience = 5
    trigger = 0
    acc_dict = {}

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        for phase in ["train", "val"]:
            classifier.train() if phase == "train" else classifier.eval()

            running_loss = 0
            all_logits = []
            all_preds = []
            all_labels = []
            num_batches = 0

            for batch in dataloaders_dict[phase]:
                optimizer.zero_grad()

                modalities_data, modalities_type, mask_rad, mask_ds, ts_lens, labels = (
                    picme_utils.prep_data(args.modalities, batch, device)
                )

                # Freeze/Activate Contrastive Weights for Each Interation
                if args.freeze:
                    model_contrastive.eval()
                elif phase == "train" and not args.freeze:
                    model_contrastive.train()

                # Retrieve embeddings for all modalities
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

                with torch.set_grad_enabled(phase == "train"):
                    if args.model_name != "baseline":
                        concatenated_embeddings = picme_utils.secure_fusion(
                            embeddings, device, args.fusion_method
                        )
                        classifier.train() if phase == "train" else classifier.eval()
                        
                        outputs = classifier(concatenated_embeddings)
                    else:
                        model_contrastive.train()
                        outputs = model_contrastive(
                            modalities_data, modalities_type, mask_rad, mask_ds, ts_lens
                        )
    
                    labels = picme_utils.extract_labels(labels, args.task).to(device)
                    labels = labels.float()
                    print(f"Outputs dtype: {outputs.dtype}")  # Should be torch.float
                    print(f"Labels dtype: {labels.dtype}")
                    outputs = outputs.to(torch.float)
                    

                    loss = criterion(outputs, labels)
                    running_loss += loss.item()

                    preds = picme_utils.predict(outputs, task=args.task)
                    all_preds.extend(preds)
                    all_labels.extend(labels.cpu().numpy())
                    all_logits.extend(outputs.cpu().detach().numpy())

                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                        
                    del outputs, loss
                    torch.cuda.empty_cache()
                    gc.collect()

                num_batches += 1

            epoch_loss = running_loss / num_batches
            epoch_metrics = picme_utils.compute_epoch_metrics(
                args.metrics, args.task, all_labels, all_preds, all_logits, phase
            )
            epoch_metrics[f"{phase}_loss"] = epoch_loss
            epoch_out = f"Epoch: {epoch}, Loss: {epoch_loss:.4f}, "
            for metric in args.metrics:
                metric_name = f"{phase}_{metric}"
                epoch_out += f"{metric}: {epoch_metrics[metric_name]}, "
            print(epoch_out)

            wandb.log(epoch_metrics)
            epoch_objective = epoch_metrics[f"{phase}_{args.objective}"]
            if phase == "val":
                acc_dict[epoch] = epoch_objective
                if args.model_name == "baseline":
                    torch.save(model_contrastive.state_dict(), path + "_current.pth")
                else:
                    torch.save(classifier.state_dict(), path + "_current.pth")

                if epoch_objective > best_metric:
                    best_metric = epoch_objective
                    if args.model_name == "baseline":
                        torch.save(
                            model_contrastive.state_dict(),
                            path + f"_{args.objective}_best.pth",
                        )
                    else:
                        torch.save(
                            classifier.state_dict(),
                            path + f"_{args.objective}_best.pth",
                        )

                if (epoch > 10) and (acc_dict[epoch] <= acc_dict[epoch - 10]):
                    trigger += 1
                    if trigger >= patience:
                        if args.model_name == "baseline":
                            return model_contrastive
                        else:
                            return classifier
                else:
                    trigger = 0

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_metric))

    if args.model_name == "baseline":
        return model_contrastive
    else:
        return classifier


def main(args: argparse.Namespace):
    modalities = args.modalities
    task = args.task
    task = (
        args.task if args.task not in _RENAME_MODALITY else _RENAME_MODALITY[args.task]
    )
    print(f"Fine-Tuning for task {task}!")

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

    if args.freeze:
        print("Freezing contrastively learned weights")
        for param in model_contrastive.parameters():
            param.requires_grad = False

    sweep_config = {"method": "grid"}
    metric = {"name": args.objective, "goal": "minimize"}
    sweep_config["metric"] = metric

    sweep_seeds = [15, 0, 1, 67, 128, 87, 261, 510, 340, 22] #15, 0, 1, 67, 128, 87, 261, 510, 340, 22
    seeds = [args.seed_number] if args.seed_number else sweep_seeds

    parameters_dict = {
        "optimizer": {"values": ["adamW"]},
        "learning_rate": {"values": args.learning_rate},
        "random_seed": {"values": seeds},
        "batch_size": {"values": args.batch_size},
        "epochs": {"values": [args.epochs]},
        "frozen": {"values": [f"{'frozen' if args.freeze else 'unfrozen'}"]},
        "ce_loss": {"values": [f"{'weighted' if args.weigh_loss else 'unweighted'}"]},
        "fusion_method": {"values": [args.fusion_method]},
        "modalities": {"values": [tuple(modalities)]},
    }

    sweep_config["parameters"] = parameters_dict
    pprint.pprint(sweep_config)
    sweep_id = wandb.sweep(sweep_config, project=args.wandb_project)

    def main_train(config=None):
        with wandb.init(config=config):
            config = wandb.config

            train_input_list, val_input_list, _ = data.get_dataloaders(
                modalities, config.batch_size, _DATA_DIR, task
            )

            dataloaders_dict = {
                "train": data.MergeLoader(train_input_list),
                "val": data.MergeLoader(val_input_list),
            }

            path = (
                args.save_prefix
                + str(args.model_name)
                + "_"
                + str(task)
                + "_"
                + str(args.fusion_method)
                + "_"
                + "_".join(modalities)
                + "_"
                + str(config.learning_rate)
                + "_"
                + str(config.batch_size)
                + "_"
                + str(config.epochs)
                + "_"
                + str(config.random_seed)
                + "_"
                + f"{'frozen' if args.freeze else 'unfrozen'}_"
                + f"{'weighted' if args.weigh_loss else 'unweighted'}"
            )

            if args.weigh_loss:
                print("Weighing loss!")
                weights = torch.Tensor(_TASK_WEIGHTS[task])
                weights = weights / max(weights)
                weights = 1 / weights
                weights = weights.to(device)
                weights = weights.to(device).float()
                criterion = nn.CrossEntropyLoss(weight=weights)
            else:
                print("Not weighing loss!")
                criterion = nn.CrossEntropyLoss()

            train_model(
                dataloaders_dict,
                criterion,
                model_contrastive,
                config,
                path,
                args,
                device,
            )

    wandb.agent(sweep_id, main_train, count=200)


if __name__ == "__main__":
    args = finetune_arg_parser()
    main(args)
