# read data
import pandas as pd
import torch
from datasets import DatasetDict, Dataset
from copy import deepcopy
import numpy as np

def load_gtc(filename: str, format="transformers", filter_role="Witness", sn_train = int(18854*0.8), seed=1):
    """ format: string describing for which library the dataset should be prepared. options =  {"transformers", "lists"}
        seed: The seed to split into train and test.
    """
    data = pd.read_csv(filename, sep=";")
    if filter_role is not None:
        data = data[data["role"]==filter_role]
    
    X_data = data['text'].to_numpy()
    Y_data = data['trauma'].to_numpy()
    #tribunals = data['tribunal'].to_numpy()
    
    ## Simple train test split


    n_train = sn_train
    randperm = np.random.default_rng(seed=seed).permutation(len(Y_data))
    X_train = X_data[randperm[:n_train]]
    y_train = Y_data[randperm[:n_train]]
    X_test = X_data[randperm[n_train:]]
    y_test = Y_data[randperm[n_train:]]

    if format=="transformers":
        trainset = Dataset.from_dict({"text": X_train, "label": torch.tensor(y_train, dtype=torch.long)})
        testset = Dataset.from_dict({"text": X_test, "label":  torch.tensor(y_test, dtype=torch.long)})
        genocide_ds = DatasetDict({"train": trainset, "test": testset})
        return genocide_ds
    elif format=="lists":
        trainset = (X_train, y_train)
        testset = (X_test, y_test)
        return {"train": trainset, "test": testset}
    else:
        raise ValueError(f"Unknown format {format}")

def load_ptsd(filename: str, format="transformers", seed=1, n_train=1000):
    """ Load PTSD dataset """
    data = pd.read_csv(filename)
    randperm = np.random.default_rng(seed=seed).permutation(len(data))
    X_data = data["text"].to_numpy()
    Y_data = data["majority_vote"].to_numpy()

    X_train = X_data[randperm[:n_train]]
    y_train = Y_data[randperm[:n_train]]
    X_test = X_data[randperm[n_train:]]
    y_test = Y_data[randperm[n_train:]]

    if format=="transformers":
        trainset = Dataset.from_dict({"text": X_train, "label": torch.tensor(y_train, dtype=torch.long)})
        testset = Dataset.from_dict({"text": X_test, "label":  torch.tensor(y_test, dtype=torch.long)})
        genocide_ds = DatasetDict({"train": trainset, "test": testset})
        return genocide_ds
    elif format=="lists":
        trainset = (X_train, y_train)
        testset = (X_test, y_test)
        return {"train": trainset, "test": testset}
    else:
        raise ValueError(f"Unknown format {format}")


def load_counsel(filename: str, format="transformers", seed=1, n_train=1000):
    """ Load counseling dataset """
    data = pd.read_csv(filename)
    randperm = np.random.default_rng(seed=seed).permutation(len(data))
    X_data = data["text"].to_numpy()
    Y_data = data["majority_vote"].to_numpy()

    X_train = X_data[randperm[:n_train]]
    y_train = Y_data[randperm[:n_train]]
    X_test = X_data[randperm[n_train:]]
    y_test = Y_data[randperm[n_train:]]

    if format=="transformers":
        trainset = Dataset.from_dict({"text": X_train, "label": torch.tensor(y_train, dtype=torch.long)})
        testset = Dataset.from_dict({"text": X_test, "label":  torch.tensor(y_test, dtype=torch.long)})
        genocide_ds = DatasetDict({"train": trainset, "test": testset})
        return genocide_ds
    elif format=="lists":
        trainset = (X_train, y_train)
        testset = (X_test, y_test)
        return {"train": trainset, "test": testset}
    else:
        raise ValueError(f"Unknown format {format}")

def load_incels(filename: str, format="transformers", seed=1):
    """ Load incel dataset, note that this dataset has only a test partition. """
    data = pd.read_csv(filename)
    randperm = np.random.default_rng(seed=seed).permutation(len(data))
    X_data = data["text"].to_numpy()
    Y_data = data["majority_vote"].to_numpy()

    if format=="transformers":
        testset = Dataset.from_dict({"text": X_data, "label":  torch.tensor(Y_data, dtype=torch.long)})
        genocide_ds = DatasetDict({"test": testset})
        return genocide_ds
    elif format=="lists":
        return {"test": testset}
    else:
        raise ValueError(f"Unknown format {format}")

