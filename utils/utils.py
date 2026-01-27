'''
Utility functions
'''

import random
import numpy as np
import torch


def set_seed(seed: int) -> int:
    """
    Set global random seeds for reproducibility
    """
    
    # Set global random seeds for reproducibility
    seed_value = seed
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)


    # If using CUDA, also set the seed for CUDA operations
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # For multi-GPU operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Global random seeds set to {seed_value} for random, numpy, and torch.")
    return seed_value