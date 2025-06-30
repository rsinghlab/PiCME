import os
import random

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import (average_precision_score, f1_score, hamming_loss,
                             roc_auc_score)


def set_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def build_optimizer(network, optimizer, learning_rate, momentum):
    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
    elif optimizer == "adam":
        optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    elif optimizer == "adamW":
        optimizer = optim.AdamW(network.parameters(), lr=learning_rate, eps=1e-8)
    return optimizer


def prep_data(modalities, batch, device):
    modalities_data = []
    modalities_type = []
    mask_rad, mask_ds, ts_lengths = None, None, None
    labels = None  # Ensure labels are defined in each loop iteration

    # Assuming each batch is a list of tuples (data, optional mask) for each modality
    for modality, data in zip(modalities, batch):
        if modality == "text_rad":
            data, mask_rad, labels = data
            mask_rad = mask_rad.to(device)
        elif modality == "text_ds":
            data, mask_ds, labels = data
            mask_ds = mask_ds.to(device)
        elif modality == "ts":
            data, ts_lengths, labels = data
            ts_lengths = ts_lengths.to(device)
        else:  # This includes "img" and "demo"
            data, labels = data

        modalities_data.append(data.to(device))
        modalities_type.append(modality)

    return modalities_data, modalities_type, mask_rad, mask_ds, ts_lengths, labels


def secure_fusion(embeddings, device, fusion_method):
    safe_embeddings = []
    for embedding in embeddings:
        if embedding.dim() == 1:
            safe_embeddings.append(torch.unsqueeze(embedding, 0))
        else:
            safe_embeddings.append(embedding)

    if fusion_method == "concatenation":
        fused_embeddings = torch.cat(safe_embeddings, dim=1).to(device)
    elif fusion_method == "vanilla_lstm" or fusion_method == "modality_lstm":
        expanded_embeddings = []
        for embedding in safe_embeddings:
            expanded_embeddings.append(embedding[:, None, :])
        fused_embeddings = torch.cat(expanded_embeddings, dim=1)

    return fused_embeddings


def extract_labels(raw_labels, task):
    if task == "mortality":
        return raw_labels.long()
    elif task == "phenotyping":
        return raw_labels


def predict(outputs, task):
    if task == "mortality":
        _, preds = torch.max(outputs, axis=1)
        return preds.cpu().detach().numpy()
    elif task == "phenotyping":
        return (outputs.sigmoid() > 0.5).cpu().detach().numpy()
    else:
        return ValueError("Invalid task given.")


def compute_epoch_metrics(metrics, task, all_labels, all_preds, all_logits, phase):
    epoch_metrics = dict()
    
    all_labels, all_preds, all_logits = (
        np.asarray(all_labels),
        np.asarray(all_preds),
        np.asarray(all_logits),
    )

    if task == "mortality":
        print("reshaping mortality")
        all_logits = all_logits[:, 1]
    
    for metric in metrics:
        if metric == "hamming":  # only supported for task != mortality
            epoch_metrics[f"{phase}_hamming"] = hamming_loss(all_labels, all_preds)
        elif metric == "auprc":
            epoch_metrics[f"{phase}_auprc"] = average_precision_score(
                all_labels, all_logits, average="macro"
            )
        elif metric == "auroc":
            epoch_metrics[f"{phase}_auroc"] = roc_auc_score(
                all_labels, all_logits, average="macro"
            )
        elif metric == "f1":
            epoch_metrics[f"{phase}_f1"] = f1_score(
                all_labels, all_preds, average="weighted"
            )

    if task == "phenotyping":
        epoch_metrics[f"{phase}_accuracy"] = np.mean(
            (np.sum((np.array(all_labels) == np.array(all_preds)), axis=1) / 25)
        )
    elif task == "mortality":
        epoch_metrics[f"{phase}_accuracy"] = sum(all_preds == all_labels) / len(
            all_labels
        )

    return epoch_metrics


def compute_accuracy(image_output, text_output):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Compute the similarity score matrix
    scores = torch.matmul(image_output, text_output.t())

    # Get the indices that would sort the scores
    sorted_indices = torch.argsort(scores, dim=1, descending=True)

    # Get the rank of the correct text for each image
    targets = torch.arange(image_output.size(0)).to(device)
    ranks = (sorted_indices == targets.view(-1, 1)).nonzero()[:, 1]

    top1 = torch.sum(ranks < 1).item() / image_output.size(0)
    top5 = torch.sum(ranks < 5).item() / image_output.size(0)

    return top1, top5


def compute_multimodal_accuracy(*embeddings):
    n_modalities = len(embeddings)
    batch_size = embeddings[0].size(0)

    # Create a similarity score matrix for every pair of modalities
    score_matrices = [
        torch.matmul(embeddings[i], embeddings[j].t())
        for i in range(n_modalities)
        for j in range(n_modalities)
        if i != j
    ]

    targets = torch.arange(batch_size).to(embeddings[0].device)

    top1s, top5s, top10s = [], [], []  # Add list for top10s
    for scores in score_matrices:
        # Get the indices that would sort the scores
        sorted_indices = torch.argsort(scores, dim=1, descending=True)

        # Get the rank of the correct pairing
        ranks = (sorted_indices == targets.view(-1, 1)).nonzero(as_tuple=True)[1]

        top1 = torch.sum(ranks < 1).item() / batch_size
        top5 = torch.sum(ranks < 5).item() / batch_size
        top10 = torch.sum(ranks < 10).item() / batch_size  # Compute top10

        top1s.append(top1)
        top5s.append(top5)
        top10s.append(top10)  # Append top10 result to list

    avg_top1 = sum(top1s) / len(top1s)
    avg_top5 = sum(top5s) / len(top5s)
    avg_top10 = sum(top10s) / len(top10s)  # Calculate average top10

    return avg_top1, avg_top5, avg_top10  # Return top10 as well
