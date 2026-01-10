import argparse
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report
from torch import nn
from collections import defaultdict
from torch import cuda
from tqdm import tqdm
from datetime import datetime
import os 
from data_loader import Dataset, DataLoader
import train_config
# Filter the warning.
import warnings

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)
if __name__ == "__main__":
    # Get the config file
    config = train_config.config
    config["n_epochs"] =1# config['n_lin_epoch'] + config['n_dec_epoch']
    # Set Path for data
    root = config["data_path"]
