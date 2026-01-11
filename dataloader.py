# brain_dataset.py
"""
Custom PyTorch Dataset and DataLoaders for the neuroimaging toy dataset.
Assumptions:
- label.mat contains 'label' variable (shape: 500 or 500×1)
- Feature files: s1_feature.mat, s2_feature.mat, ..., s500_feature.mat
  Inside: 'F' → (400, 1632) float array
- Cluster files: s1_cluster_index.mat, s2_cluster_index.mat, ..., s500_cluster_index.mat
  Inside: 'C' → (45, 54, 45) integer array
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
from sklearn.model_selection import train_test_split
from typing import Optional, Dict, Any
# import train_config


class BrainROIDataset(Dataset):
    """
    Loads per-subject ROI feature matrices (400×1632) and optionally cluster maps.
    Labels are loaded once from label.mat and indexed by subject number (1-based).
    """
    def __init__(
        self,
        subject_ids: list[int],
        label_data_dir: str,
        feature_data_dir: str,
        cluster_data_dir: str,
        load_cluster: bool = False,
        feature_key: str = 'F',
        cluster_key: str = 'C'
    ):
        """
        Args:
            subject_ids: List of subject IDs (integers 1 to 500)
            data_dir: Root directory containing all .mat files
            load_cluster: Whether to load the 3D cluster maps (memory intensive)
            feature_key: Key name for feature matrix in .mat file
            cluster_key: Key name for cluster index in .mat file
        """
        self.subject_ids = sorted(subject_ids)
        self.data_dir = label_data_dir
        self.feature_data=feature_data_dir
        self.cluster_data=cluster_data_dir
        self.load_cluster = load_cluster
        self.feature_key = feature_key
        self.cluster_key = cluster_key

        # Load global labels once
        label_path = self.data_dir
        # print(label_path)
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")

        labels_mat = sio.loadmat(label_path)
        self.labels = labels_mat['label'].flatten().astype(np.int64)  # shape (500,)

        if len(self.labels) != 500:
            raise ValueError(f"Expected 500 labels, got {len(self.labels)}")

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sid = self.subject_ids[idx]  # 1-based subject ID

        # 1. Load feature matrix
        feature_path = os.path.join(self.feature_data, f"s_{sid}_feature.npy")
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Feature file missing: {feature_path}")

        feat_data = np.load(feature_path,allow_pickle=True)
        features=feat_data.item()['feature_mat']
        if features.shape[0]!=400:
            D=features.shape[1]
            zero_row=np.zeros((1,D),dtype=features.dtype)
            features=np.vstack([F, zero_row])
        # features = feat_data[self.feature_key].astype(np.float32)  # (400, 1632)
        
        cluster_path=os.path.join(self.cluster_data, f"s_{sid}_cluster_index.npy")
        if not os.path.exists(cluster_path):
            raise FileNotFoundError(f"Cluster file missing: {cluster_path}")
        clus_data=np.load(cluster_path, allow_pickle=True)
        clusters_matrix=clus_data.item()['cluster_index_mat']

        # 2. Optional: Load cluster map
        # cluster = None
        # if self.load_cluster:
        #     cluster_path = os.path.join(self.data_dir, f"s{sid}_cluster_index.mat")
        #     if not os.path.exists(cluster_path):
        #         raise FileNotFoundError(f"Cluster file missing: {cluster_path}")

        #     clust_data = sio.loadmat(cluster_path)
        #     cluster = clust_data[self.cluster_key].astype(np.int16)  # (45,54,45)

        # 3. Get corresponding label (0-based index in array)
        label = self.labels[sid - 1]

        # Prepare output
        item = {
            'features': torch.from_numpy(features),     # [400, 1632]
            'cluster_map' : torch.from_numpy(clusters_matrix),
            'label': torch.tensor(label, dtype=torch.long),
            'subject_id': sid
        }

        # if cluster is not None:
        #     item['cluster_map'] = torch.from_numpy(cluster)

        return item


def create_dataloaders(
    label_data_dir,
    feature_data_dir,
    cluster_data_dir,
    batch_size: int = 16,
    num_workers: int = 0,
    pin_memory: bool = True,
    load_cluster: bool = False,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates train/val/test DataLoaders with standard split (70/15/15 by default)

    Returns:
        (train_loader, val_loader, test_loader)
    """
    exclude={4, 6, 173, 175, 211, 232, 293, 319, 344, 378, 381, 391, 427, 461}
    all_subjects = [i for i in range(1, 501) if i not in exclude]  # 1 to 500

    # Stratified split (though binary labels, still good practice)
    train_subjs, temp_subjs = train_test_split(
        all_subjects,
        test_size=(val_size + test_size),
        random_state=random_state
    )

    val_subjs, test_subjs = train_test_split(
        temp_subjs,
        test_size=test_size / (val_size + test_size),
        random_state=random_state
    )

    print(f"Dataset split → Train: {len(train_subjs)} | Val: {len(val_subjs)} | Test: {len(test_subjs)}")
    train_ds = BrainROIDataset(train_subjs, label_data_dir, feature_data_dir, cluster_data_dir, load_cluster=load_cluster)
    val_ds   = BrainROIDataset(val_subjs,   label_data_dir, feature_data_dir, cluster_data_dir, load_cluster=load_cluster)
    test_ds  = BrainROIDataset(test_subjs,  label_data_dir, feature_data_dir, cluster_data_dir, load_cluster=load_cluster)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader

