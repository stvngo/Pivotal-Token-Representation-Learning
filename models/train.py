'''
Define and train linear probe using the curated dataset
'''

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.nn.functional import softmax
from torch.nn.functional import cross_entropy
from torch.nn.functional import binary_cross_entropy
from torch.nn.functional import binary_cross_entropy_with_logits

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from utils.utils import set_seed
import time
import os
import logging

def train_model(model: torch.nn.Module, dataset: list[dict], epochs: int) -> None:
    """
    Train the linear probe on the Qwen model's hidden states
    """
    return None # TODO: Implement model training

def verify_model(model: torch.nn.Module) -> bool:
    return True # TODO: Implement model verification

def save_model(model: torch.nn.Module, path: str) -> None:
    return None # TODO: Implement model saving