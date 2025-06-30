import logging
import pickle
import pprint
import sys
import time

import torch
import torch.nn as nn
import wandb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)

from model_utils import *
from models import Fusion

print("PyTorch Version: ", torch.__version__)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

torch.cuda.empty_cache()


def train_model(
    dataloaders_dict,
    len_train,
    len_val,
    criterion,
    config,
    path,
    modality_shapes,
    task_classes,
):
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    set_seed(config.random_seed)

    projection_dim = 256

    classification_model = Fusion(
        projection_dim, task_classes, modality_shapes["ts"], config
    )
    classification_model.to(device)

    num_epochs = config.epochs
    optimizer = build_optimizer(
        classification_model, config.optimizer, config.learning_rate, 0.01
    )

    since = time.time()

    best_acc = 0.0
    patience = 5
    trigger = 0
    acc_dict = {}

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                length = len_train
                classification_model.train()
            else:
                length = len_val
                classification_model.eval()

            running_loss = 0.0
            all_preds = []
            all_logits = []
            all_labels = []

            num_batches = 0
            for batch in dataloaders_dict[phase]:
                optimizer.zero_grad()
                tx_data, tx_mask_ds = None, None
                data_ts, mask_ts, delta_ts, lengths_ts = None, None, None, None
                labels = None  # Ensure labels are defined in each loop iteration

                # Assuming each batch is a list of tuples (data, optional mask) for each modality
                modalities = ["img", "ts"]
                for modality, data in zip(modalities, batch):
                    if modality == "ts":
                        data_ts, ts_lengths, labels = data
                        ts_lengths = ts_lengths.to(device)
                    elif modality == "img":
                        data_img, labels = data

                with torch.set_grad_enabled(phase == "train"):
                    if phase == "train":
                        classification_model.train()
                    else:
                        classification_model.eval()

                    fusion_results = classification_model(data_ts, data_img, ts_lengths)
                    logits = fusion_results["lstm"]
                    labels = labels.long().to(device)

                    # Compute loss
                    loss = criterion(logits, labels)
                    running_loss += loss.item()

                    logits = logits.cpu().detach().numpy()
                    all_logits.extend(logits[:, 1])
                    all_preds.extend(np.argmax(logits, axis=1))
                    all_labels.extend(labels.cpu().numpy())

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                num_batches += 1

            # Compute metrics after the epoch
            epoch_loss = running_loss / num_batches
            epoch_accuracy = accuracy_score(all_labels, all_preds)
            epoch_auroc = roc_auc_score(all_labels, all_logits, average="macro")
            epoch_f1_score = f1_score(all_labels, all_preds, average="weighted")

            print(
                f"Epoch: {epoch}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, F1-Score: {epoch_f1_score:.4f}, AUROC: {epoch_auroc:.4f}"
            )
            if phase == "val":
                wandb.log(
                    {
                        "val_loss": epoch_loss,
                        "val_acc": epoch_accuracy,
                        "val_auroc": epoch_auroc,
                        "val_f1": epoch_f1_score,
                    }
                )
                acc_dict[epoch] = epoch_f1_score
                torch.save(classification_model.state_dict(), path + "_current.pth")
                if epoch_f1_score > best_acc:
                    best_acc = epoch_f1_score
                    torch.save(classification_model.state_dict(), path + "_best.pth")

                if (epoch > 10) and (acc_dict[epoch] <= acc_dict[epoch - 10]):
                    trigger += 1
                    if trigger >= patience:
                        return classification_model
                else:
                    trigger = 0
            if phase == "train":
                wandb.log(
                    {
                        "train_loss": epoch_loss,
                        "train_acc": epoch_accuracy,
                        "train_f1": epoch_f1_score,
                        "epoch": epoch,
                        "train_auroc": epoch_auroc,
                    }
                )

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))

    return classification_model


def main():
    modalities = ["img", "ts"]
    ehr_data_dir = (
        "<your mimic data dir>"
    )
    task = "in-hospital-mortality"

    if task == "in-hospital-mortality":
        num_classes = 2
    elif task == "phenotyping":
        num_classes = 27

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    modality_shapes = {"ts": 76}
    sweep_config = {"method": "grid"}
    metric = {"name": "val_loss", "goal": "minimize"}
    sweep_config["metric"] = metric

    parameters_dict = {
        "optimizer": {"values": ["adamW"]},
        "ts_layers": {"values": [1]},
        "learning_rate": {"values": [0.0001, 0.00001, 0.00005, 0.000001, 0.000005]},
        "random_seed": {"values": [87, 261, 510, 340, 22]},  # 42, 15, 0, 1, 67, 128
        "batch_size": {"values": [16, 32]},
        "epochs": {"values": [100]},
    }

    sweep_config["parameters"] = parameters_dict

    pprint.pprint(sweep_config)

    project_name = "MedFuse-HPSweep"

    sweep_id = wandb.sweep(sweep_config, project=project_name)

    def main_train(config=None):
        with wandb.init(config=config):
            config = wandb.config

            train_input_list = []
            val_input_list = []
            for modality_name in modalities:
                train_inputs = load_dataset(
                    f"{ehr_data_dir}/{task}/train_finetune_{modality_name}_dataset.pkl"
                )
                val_inputs = load_dataset(
                    f"{ehr_data_dir}/{task}/val_finetune_{modality_name}_dataset.pkl"
                )

                if modality_name == "ts":
                    train_input_list.append(
                        DataLoader(
                            train_inputs,
                            batch_size=config.batch_size,
                            collate_fn=collate_IHM,
                            shuffle=False,
                        )
                    )
                    val_input_list.append(
                        DataLoader(
                            val_inputs,
                            batch_size=config.batch_size,
                            collate_fn=collate_IHM,
                            shuffle=False,
                        )
                    )
                else:
                    train_input_list.append(
                        DataLoader(
                            train_inputs, batch_size=config.batch_size, shuffle=False
                        )
                    )
                    val_input_list.append(
                        DataLoader(
                            val_inputs, batch_size=config.batch_size, shuffle=False
                        )
                    )

            len_val = len(val_input_list[0].dataset)
            len_train = len(train_input_list[0].dataset)

            dataloaders_dict = {
                "train": MyLoader(train_input_list),
                "val": MyLoader(val_input_list),
            }

            path = (
                "models/medfuse_"
                + str(task)
                + "_"
                + str(config.learning_rate)
                + "_"
                + str(config.random_seed)
                + "_"
                + str(config.batch_size)
                + "_"
                + str(config.epochs)
            )

            # IHM: [8377, 1357]
            n_samples = torch.tensor([8377, 1357], dtype=torch.float)
            weights = 1.0 / n_samples
            weights / min(weights)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            weights = weights.to(device)
            criterion = nn.CrossEntropyLoss()

            train_model(
                dataloaders_dict,
                len_train,
                len_val,
                criterion,
                config,
                path,
                modality_shapes,
                num_classes,
            )

    wandb.agent(sweep_id, main_train, count=200)


if __name__ == "__main__":
    main()
