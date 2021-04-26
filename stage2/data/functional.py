from scipy.stats import norm
import numpy as np

import torch
from torch.utils.data import Dataset, Subset

from typing import Union


def age_norm(x, mu, sigma=15):
    z = (x - mu) / sigma
    return norm.pdf(z) / norm.pdf(0)


def train_valid_split(
    dataset: Dataset, 
    valid_ratio: float = 0.2,
    shuffle: bool = True
):    
    data_size = len(dataset)
    valid_size = int(data_size * valid_ratio)
    
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    
    indices_train, indices_valid = indices[valid_size:], indices[:valid_size]
    train, valid = Subset(dataset, indices_train), Subset(dataset, indices_valid)

    return train, valid


def train_valid_raw_split(
    data: Union[list, tuple, np.ndarray, torch.Tensor], 
    valid_ratio: float = 0.2, 
    shuffle: bool = True
):
    data_size = len(data)
    valid_size = int(data_size * valid_ratio)
    
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    
    indices_train, indices_valid = indices[valid_size:], indices[:valid_size]

    data = np.array(data)
    train, valid = data[indices_train], data[indices_valid]

    return train, valid
