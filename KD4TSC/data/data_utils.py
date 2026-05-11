""" Data loading and preprocessing utilities."""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

class TSDataset(Dataset):
    """Time Series Dataset for PyTorch."""
    
    def __init__(self, X, y, transform=None):
        """
        Args:
            X: Time series data (n_samples, seq_length, n_features)
            y: Labels (n_samples, ) or one-hot (n_samples, n_classes)
            transform: Optional transform to apply
        """
        self.X = torch.FloatTensor(X)
        
        # Handle labels
        if len(y.shape) > 1: # ONe-hot encoded
            self.y = torch.FloatTensor(y)
        else:
            self.y = torch.LongTensor(y)

        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        if self.transform:
            x = self.transform

        return x, y
        


def load_ucr_dataset(root_dir, dataset_name):
    dataset_path = os.path.join(root_dir, dataset_name)
    print(f"dataset_path: {dataset_path}")
    # folder_path = "/home/jabdullayev/Codes/UCRArchive_2018/"
    # folder_path = "/home/jabdullayev/phd/datasets/UCRArchive_2018/"
    # folder_path += file_name + "/"


    train_path = os.path.join(dataset_path, f'{dataset_name}_TRAIN.tsv')
    test_path = os.path.join(dataset_path, f'{dataset_name}_TEST.tsv')

    print(f'train_path: {train_path}')
    print(f'test_path: {test_path}')
    
    if os.path.exists(test_path) <= 0:
        print("File not found")
        return None, None, None, None

    train = np.loadtxt(train_path, dtype=np.float64)
    test = np.loadtxt(test_path, dtype=np.float64)

    ytrain = train[:, 0]
    ytest = test[:, 0]

    xtrain = np.delete(train, 0, axis=1)
    xtest = np.delete(test, 0, axis=1)

    nb_classes = len(np.unique(ytrain))

    return xtrain, ytrain, xtest, ytest, nb_classes


def znormalize(x):

    stds = np.std(x, axis=1, keepdims=True)
    if len(stds[stds == 0.0]) > 0:
        stds[stds == 0.0] = 1.0
        return (x - x.mean(axis=1, keepdims=True)) / stds
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True))


def encode_labels(y):

    labenc = LabelEncoder()

    return labenc.fit_transform(y)



def create_data_loaders(x_train, y_train, x_test, y_test, batch_size=64,
                        num_workers=0, pin_memory=True,):

    x_train = znormalize(x_train)
    x_train = np.expand_dims(x_train, axis=1)

    x_test = znormalize(x_test)
    x_test = np.expand_dims(x_test, axis=1)


    y_train = encode_labels(y_train)
    y_test = encode_labels(y_test)

    train_dataset = TSDataset(x_train, y_train)
    test_dataset = TSDataset(x_test, y_test)

    # data, target = torch.from_numpy(data), torch.from_numpy(target)

    torch.manual_seed(42)    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )


    return train_loader,test_loader
