import os
import pickle
import random

import h5py
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset

ImageFile.LOAD_TRUNCATED_IMAGES = True

_MORTALITY_CLASS_WEIGHTS = [8377, 1357]
_PHENO_CLASS_WEIGHTS = [
    4899,
    1516,
    1094,
    4746,
    2766,
    1739,
    4387,
    1158,
    3717,
    3800,
    1841,
    2286,
    4463,
    5531,
    6740,
    983,
    2442,
    2368,
    1994,
    723,
    1809,
    3001,
    5263,
    3785,
    3307,
]


class TextDataset(Dataset):
    def __init__(self, dataframe, target_cols, tokenizer, type_text, max_token_len=512):
        self.tokenizer = tokenizer
        self.type_text = type_text
        self.dataframe = dataframe
        self.labels = dataframe[target_cols].values
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        data_row = self.dataframe.iloc[idx]

        text_data = data_row[self.type_text]

        encoding = self.tokenizer.encode_plus(
            text_data,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text = encoding["input_ids"].flatten()
        att_mask = encoding["attention_mask"].flatten()
        labels = torch.tensor(self.labels[idx], dtype=torch.float)
        return text, att_mask, labels


class TimeSeriesDataset(Dataset):
    def __init__(self, time_series_data):
        self.time_series_data = time_series_data

    def __len__(self):
        return len(self.time_series_data)

    def __getitem__(self, idx):
        time_series = self.time_series_data[idx]
        ts = torch.tensor(time_series, dtype=torch.float)
        return ts


class DemographicsDataset(Dataset):
    def __init__(self, dataframe, labels):
        self.features = (
            dataframe.values
        )  # .drop(labels, axis=1).values  # Assuming labels are not part of your features dataframe
        self.labels = labels  # Directly using the passed list of labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        demo = torch.tensor(self.features[idx], dtype=torch.float)
        labels = torch.tensor(self.labels[idx], dtype=torch.float)
        return demo, labels


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


class EHRdataset(Dataset):
    def __init__(self, listfile, preprocessed_dir):
        self.data_files = []
        # self.labels = []

        df = pd.read_csv(listfile)
        for idx, row in df.iterrows():
            # Load preprocessed data
            preprocessed_file = os.path.join(
                preprocessed_dir,
                os.path.basename(row["filename"]).replace(".csv", ".h5"),
            )
            if os.path.exists(preprocessed_file):
                self.data_files.append(preprocessed_file)
                # self.labels.append(row['labels'])  # Adjust based on your actual label handling

    def __getitem__(self, index):
        with h5py.File(self.data_files[index], "r") as hf:
            data = hf["data"][:]
        # label = self.labels[index]  # Adjust this based on how you handle labels
        return data  # , label

    def __len__(self):
        return len(self.data_files)


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
                self.labels.append(row[classes].values)

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
                self.labels.append(
                    row["y_true"]
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


def load_dataset(file_path):
    with open(file_path, "rb") as file:
        dataset = pickle.load(file)
    return dataset


def seq_collate(batch):
    data = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    data = [torch.tensor(d, dtype=torch.float) for d in data]

    data_padded = pad_sequence(data, batch_first=True, padding_value=0.0)
    lengths = torch.tensor([len(x) for x in data], dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    return data_padded, lengths, labels


def shuffle_indices(dataset, seed):
    indices = list(range(len(dataset)))
    random.seed(seed)
    random.shuffle(indices)
    return indices


def get_shuffled_subset(dataset, indices):
    return Subset(dataset, indices)


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


class MergeLoader(object):
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


def get_dataloaders(modalities, batch_size, data_dir, task, test=False):
    train_inputs = []
    val_inputs = []
    test_inputs = []

    # Load the first modality to get the length of the datasets
    t_inputs = load_dataset(
        f"{data_dir}/{task}/train_finetune_{modalities[0]}_dataset.pkl"
    )
    v_inputs = load_dataset(
        f"{data_dir}/{task}/val_finetune_{modalities[0]}_dataset.pkl"
    )

    # Shuffle indices once
    seed = 42  # Set a seed for reproducibility
    train_indices = shuffle_indices(t_inputs, seed)

    for modality_name in modalities:
        if test:
            test_input = load_dataset(
                f"{data_dir}/{task}/test_finetune_{modality_name}_dataset.pkl"
            )
            if modality_name == "ts":
                curr_test = DataLoader(
                    test_input,
                    batch_size=batch_size,
                    collate_fn=seq_collate,
                    shuffle=False,
                )
            else:
                curr_test = DataLoader(test_input, batch_size=batch_size, shuffle=False)
            test_inputs.append(curr_test)
        else:
            t_inputs = load_dataset(
                f"{data_dir}/{task}/train_finetune_{modality_name}_dataset.pkl"
            )
            v_inputs = load_dataset(
                f"{data_dir}/{task}/val_finetune_{modality_name}_dataset.pkl"
            )

            t_inputs = get_shuffled_subset(t_inputs, train_indices)

            if modality_name == "ts":
                curr_t = DataLoader(
                    t_inputs,
                    batch_size=batch_size,
                    collate_fn=seq_collate,
                    shuffle=False,
                )
                curr_v = DataLoader(
                    v_inputs,
                    batch_size=batch_size,
                    collate_fn=seq_collate,
                    shuffle=False,
                )
            else:
                curr_t = DataLoader(t_inputs, batch_size=batch_size, shuffle=False)
                curr_v = DataLoader(v_inputs, batch_size=batch_size, shuffle=False)

            train_inputs.append(curr_t)
            val_inputs.append(curr_v)

    if test:
        return None, None, test_inputs
    else:
        return train_inputs, val_inputs, None
