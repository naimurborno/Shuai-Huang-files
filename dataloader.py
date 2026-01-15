
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
from sklearn.model_selection import train_test_split, StratifiedKFold
from typing import Optional, Dict, Any
import train_config


config = train_config.config


class BRDataset(Dataset):

    def __init__(
        self,
        subject_ids: list[int],
        label_data_dir: str,
        feature_data_dir: str,
        cluster_data_dir: str,
    ):
        self.subject_ids = sorted(subject_ids)
        self.data_dir = label_data_dir
        self.feature_data=feature_data_dir
        self.cluster_data=cluster_data_dir

        # Load global labels once
        label_path = self.data_dir
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")
        
        #Load Labels
        labels_mat = sio.loadmat(label_path)
        self.labels = labels_mat['label'].flatten().astype(np.int64)  # shape (500,)

        if len(self.labels) != 500:
            raise ValueError(f"Expected 500 labels, got {len(self.labels)}")

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sid = self.subject_ids[idx]  

        # Load feature matrix
        feature_path = os.path.join(self.feature_data, f"s_{sid}_feature.npy")
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Feature file missing: {feature_path}")
        feat_data = np.load(feature_path,allow_pickle=True)
        features=feat_data.item()['feature_mat'] # shape (400,1632)

        # Load Cluster Index Matrix
        cluster_path=os.path.join(self.cluster_data, f"s_{sid}_cluster_index.npy")
        if not os.path.exists(cluster_path):
            raise FileNotFoundError(f"Cluster file missing: {cluster_path}")
        clus_data=np.load(cluster_path, allow_pickle=True)
        clusters_matrix=clus_data.item()['cluster_index_mat'] # shape (45,54,45)

        label = self.labels[sid - 1]

        # Prepare output
        item = {
            'features': torch.from_numpy(features),     # [400, 1632]
            'cluster_map' : torch.from_numpy(clusters_matrix), # [45, 54, 45]
            'label': torch.tensor(label, dtype=torch.long),
            'subject_id': sid
        }
        return item


def create_dataloaders(
    label_data_dir,
    feature_data_dir,
    cluster_data_dir,
    batch_size: int = 16,
    num_workers: int = 0,
    test_size: float = 0.15,
    random_state: int = 42,
    exclude_list: int=[]
):
    labels_mat = sio.loadmat(os.path.join(label_data_dir,'label.mat'))
    labels = labels_mat['label'].flatten().astype(np.int64)

    # exclude={4, 6, 173, 175, 211, 232, 293, 319, 344, 378, 381, 391, 427, 461}
    all_subjects =[i for i in range(1, len(labels)+1) if i not in exclude_list] # Excluding Subjects which does not have proper shape

    trainval_subjs, test_subjs = train_test_split(
        all_subjects,
        test_size=test_size,
        random_state=random_state
    )
    trainval_subjs=np.array(trainval_subjs)
    test_ds  = BRDataset(test_subjs,  label_data_dir, feature_data_dir, cluster_data_dir)   

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    skf=StratifiedKFold(n_splits=config['n_split'],shuffle=True, random_state=random_state)
    folds=[]
    trainval_labels=labels[trainval_subjs-1]
    for fold_id, (tr_idx,val_idx) in enumerate(skf.split(trainval_subjs,trainval_labels)):
        train_ids=trainval_subjs[tr_idx]
        val_ids=trainval_subjs[val_idx]

        train_ds=BRDataset(train_ids,label_data_dir,feature_data_dir, cluster_data_dir)
        val_ds=BRDataset(val_ids,label_data_dir,feature_data_dir, cluster_data_dir)

        train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
        )

        val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        )

        folds.append((train_loader,val_loader))

    return folds, test_loader

