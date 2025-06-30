import json
import logging
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from PIL import Image, ImageFile
from torch.utils.data import (DataLoader, Dataset, RandomSampler,
                              SequentialSampler)
from torchvision import datasets, models, transforms
# import pandas_path  # Path style access for pandas
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True
import pickle

import h5py
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel


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


def load_dataset(file_path):
    with open(file_path, "rb") as file:
        dataset = pickle.load(file)
    return dataset


def build_optimizer(network, optimizer, learning_rate, momentum, decay):
    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
    elif optimizer == "adam":
        optimizer = optim.Adam(
            network.parameters(), lr=learning_rate, weight_decay=decay
        )
    elif optimizer == "adamW":
        optimizer = optim.AdamW(network.parameters(), lr=learning_rate, eps=1e-8)
    return optimizer


class MedicalImageDataset(Dataset):
    def __init__(self, dataframe, target_cols, img_col, transform=None):
        self.dataframe = dataframe
        self.img_col = img_col
        self.labels = dataframe[target_cols].values
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        labels = torch.tensor(self.labels[idx], dtype=torch.float)
        img_path = self.dataframe.iloc[idx][self.img_col]
        image = Image.open(img_path).convert("RGB")  # Convert to RGB for consistency

        if self.transform:
            image = self.transform(image)

        return image, labels


class EHRdatasetFinetune(Dataset):
    def __init__(self, listfile, preprocessed_dir, classes):
        self.data_files = []
        self.labels = []

        df = pd.read_csv(listfile)
        for idx, row in df.iterrows():
            # Load preprocessed data
            preprocessed_file = os.path.join(
                preprocessed_dir, os.path.basename(row["stay"]).replace(".csv", ".h5")
            )
            if os.path.exists(preprocessed_file):
                self.data_files.append(preprocessed_file)
                self.labels.append(
                    row[classes].values
                )  # Adjust based on your actual label handling

    def __getitem__(self, index):
        try:
            with h5py.File(self.data_files[index], "r") as hf:
                data = hf["data"][:]
            label = self.labels[index]  # Adjust this based on how you handle labels
            return data, label

        except KeyError as e:
            print(f"Error loading data from file {self.data_files[index]}: {e}")
            raise

    def __len__(self):
        return len(self.data_files)


class EHRdatasetFinetuneIHM(Dataset):
    def __init__(self, listfile, preprocessed_dir):
        self.data_files = []
        self.labels = []

        df = pd.read_csv(listfile)
        for idx, row in df.iterrows():
            # Load preprocessed data
            preprocessed_file = os.path.join(
                preprocessed_dir, os.path.basename(row["stay"]).replace(".csv", ".h5")
            )
            if os.path.exists(preprocessed_file):
                self.data_files.append(preprocessed_file)
                self.labels.append(row["y_true"])

    def __getitem__(self, index):
        try:
            with h5py.File(self.data_files[index], "r") as hf:
                data = hf["data"][:]
            label = self.labels[index]  # Adjust this based on how you handle labels
            return data, label

        except KeyError as e:
            print(f"Error loading data from file {self.data_files[index]}: {e}")
            raise

    def __len__(self):
        return len(self.data_files)


def collate_IHM(batch):
    dim = np.asarray(batch[0][0]).shape[-1]

    data, labels, lengths, masks = [], [], [], []
    for item in batch:
        data.append(torch.tensor(item[0], dtype=torch.float))
        length = len(item[0])
        lengths.append(length)
        masks.append(torch.ones(length, dim))
        labels.append(item[1])

    labels = torch.tensor(labels, dtype=torch.long)
    data_padded = pad_sequence(data, batch_first=False, padding_value=0.0)
    masks_padded = pad_sequence(masks, batch_first=False, padding_value=0)
    time_stamps = (
        torch.arange(0, data_padded.size()[0])
        .unsqueeze(1)
        .unsqueeze(1)
        .expand(-1, len(batch), dim)
    )

    return data_padded, masks_padded, time_stamps, labels


class MyIter(object):
    """An iterator."""

    def __init__(self, my_loader):
        self.my_loader = my_loader
        self.loader_iters = [iter(loader) for loader in self.my_loader.loaders]

    def __iter__(self):
        return self

    def __next__(self):
        # When the shortest loader (the one with minimum number of batches)
        # terminates, this iterator will terminates.
        # The `StopIteration` raised inside that shortest loader's `__next__`
        # method will in turn gets out of this `__next__` method.
        batches = [next(loader_iter) for loader_iter in self.loader_iters]
        return self.my_loader.combine_batch(batches)

    def __len__(self):
        return len(self.my_loader)


class MyLoader(object):
    """This class wraps several pytorch DataLoader objects, allowing each time
    taking a batch from each of them and then combining these several batches
    into one. This class mimics the `for batch in loader:` interface of
    pytorch `DataLoader`.
    Args:
    loaders: a list or tuple of pytorch DataLoader objects
    """

    def __init__(self, loaders):
        self.loaders = loaders

    def __iter__(self):
        return MyIter(self)

    def __len__(self):
        return min([len(loader) for loader in self.loaders])

    # Customize the behavior of combining batches here.
    def combine_batch(self, batches):
        return batches


def my_collate(batch):
    data = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    data = [torch.tensor(d, dtype=torch.float) for d in data]

    data_padded = pad_sequence(data, batch_first=True, padding_value=0.0)
    lengths = torch.tensor([len(x) for x in data], dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    return data_padded, lengths, labels
