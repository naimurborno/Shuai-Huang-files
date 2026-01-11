import argparse
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, classification_report
from torch import nn
from collections import defaultdict
from torch import cuda
from tqdm import tqdm
from datetime import datetime
from AFT import AtlasFreeBrainTransformer
from APPLY_PCA import apply_pca
import os 
# from data_loader import Dataset, DataLoader
import train_config
from dataloader import create_dataloaders
# Filter the warning.
import warnings

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)
if __name__ == "__main__":
    # Get the config file
    config = train_config.config
    train_loader, val_loader, test_loader = create_dataloaders(
    label_data_dir=config['label_data_dir'],
    feature_data_dir=config['feature_data_dir'],
    cluster_data_dir=config['cluster_data_dir'],
    batch_size=8,
    load_cluster=False,       # ← Set True only if you really need it
)

    # Get one batch to verify
    batch = next(iter(train_loader))
    print(batch.keys())
    print("\nBatch shapes:")
    print(f"features    : {batch['features'].shape}")     # [batch, 400, 1632]
    print(f"labels      : {batch['label'].shape}")        # [batch]
    print(f"subject_ids : {batch['subject_id']}")

    if 'cluster_map' in batch:
        print(f"cluster_map : {batch['cluster_map'].shape}")

    F=apply_pca(batch['features'])    
    model = AtlasFreeBrainTransformer()
    
    # F = torch.randn(2, 400, 1632)
    # C = torch.randint(0, 401, (2, 45, 54, 45))

    logits = model(F, batch['cluster_map'])
    print(logits)  # (2, 2)
    

