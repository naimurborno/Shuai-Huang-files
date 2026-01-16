# import argparse
import numpy as np
import torch
from torch.optim import AdamW, Adam
from sklearn.metrics import accuracy_score, classification_report
from torch import nn
from collections import defaultdict
from torch import cuda, nn, optim
from tqdm import tqdm
from datetime import datetime
from AFT import AtlasFreeBrainTransformer
from torch_pca import PCA
from APPLY_PCA import apply_pca
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from utils import plot_training_curves, EarlyStopping
import os 
from torch.optim.lr_scheduler import ReduceLROnPlateau

import train_config
from dataloader import create_dataloaders
import pandas as pd
import warnings

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)

if __name__ == "__main__":
    config = train_config.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    os.makedirs(config['output_dir'], exist_ok=True)
    
    #Findding out subjects without proper shape
    exclude_list = []
    dataset_len = 0
    for i in os.listdir(config['feature_data_dir']):
        data = np.load(os.path.join(config['feature_data_dir'], i), allow_pickle=True)
        data = data.item()['feature_mat']
        dataset_len += 1
        if data.shape[0] != 400:
            exclude_list.append(int(i.split('_')[1]))
            dataset_len -= 1
    print(f"Subjects without proper shape: {exclude_list}")


    
    folds, test_loader, full_loader = create_dataloaders(
                                            label_data_dir=config['label_data_dir'],
                                            feature_data_dir=config['feature_data_dir'],
                                            cluster_data_dir=config['cluster_data_dir'],
                                            batch_size=config['batch_size'],
                                            exclude_list=exclude_list
                                        )
    
    metrics_records = []      # To store every epoch for plotting
    best_metrics_per_fold = [] # To store only the best epoch per fold for final stats
    absolute_best_val_acc = 0.0
    best_overall_model_path = ''
    best_pca_model = None

    for fold, (train_loader, val_loader) in enumerate(folds):
        print(f"\nTraining for fold: {fold+1}")
        print(".............................")
        early_stoper = EarlyStopping(patience=10)
        pca_model = PCA(n_components=config['n_components'])

        # 1. Fitting PCA model on Train Set
        all_train_features = []
        for batch in train_loader:
            features = batch['features'].to(device)
            all_train_features.append(features)
        all_train_features = torch.cat(all_train_features, dim=0)
        B, T, F = all_train_features.shape
        all_train_features = all_train_features.view(B*T, F).to(torch.float32)
        pca_model.fit(all_train_features)

        model = AtlasFreeBrainTransformer().to(device)
        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        
        fold_best_val_acc = 0.0
        fold_best_metrics = {}

        for epoch in range(config['Epochs']):
            model.train()
            correct, total, running_loss = 0, 0, 0.0
            loop = tqdm(train_loader, desc=f"Fold {fold+1} Epoch [{epoch+1}/{config['Epochs']}]")

            for batch in loop:
                features = batch['features'].to(device).to(torch.float32)
                features = apply_pca(features, pca_model=pca_model, train_data=True)
                labels = (batch['label']-1).long().to(device)
                cluster_map = batch['cluster_map'].to(device).to(torch.long)

                outputs = model(features, cluster_map)
                loss = loss_func(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                running_loss += loss.item()
                loop.set_postfix(loss=loss.item())

            train_loss = running_loss / len(train_loader)
            train_acc = 100 * correct / total

            # Validation Phase
            model.eval()
            y_true, y_pred, y_prob = [], [], []
            val_running_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    features = batch['features'].to(device).to(torch.float32)
                    features = apply_pca(features, pca_model=pca_model, train_data=False)
                    labels = (batch['label']-1).long().to(device)
                    cluster_map = batch['cluster_map'].to(device).to(torch.long)

                    outputs = model(features, cluster_map)
                    probs = torch.softmax(outputs, dim=1)[:, 1]
                    loss = loss_func(outputs, labels)
                    
                    _, predicted = torch.max(outputs.data, 1)
                    val_running_loss += loss.item()
                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(predicted.cpu().numpy())
                    y_prob.extend(probs.cpu().numpy())

            val_acc = accuracy_score(y_true, y_pred) * 100
            val_loss = val_running_loss / len(val_loader)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            sens = tp/(tp+fn) if (tp+fn) > 0 else 0
            spec = tn/(tn+fp) if (tn+fp) > 0 else 0
            auroc = roc_auc_score(y_true, y_prob)

            current_metrics = {
                "fold": fold+1, "epoch": epoch+1, "train_loss": train_loss,
                "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc,
                "val_sensitivity": sens, "val_specificity": spec, "val_AUROC": auroc
            }
            metrics_records.append(current_metrics)

            # Check if this is the best epoch for the CURRENT fold
            improved = early_stoper.step(val_acc)
            if val_acc > fold_best_val_acc:
                fold_best_val_acc = val_acc
                fold_best_metrics = current_metrics
                fold_model_path = os.path.join(config['output_dir'], f"best_fold_{fold+1}.pt")
                torch.save(model.state_dict(), fold_model_path)
                
                # Check if this is the best model OVERALL folds for the final test set
                if val_acc > absolute_best_val_acc:
                    absolute_best_val_acc = val_acc
                    best_overall_model_path = fold_model_path
                    best_pca_model = pca_model

            if early_stoper.stop:
                print("Early stopping triggered for this fold.")
                break

        best_metrics_per_fold.append(fold_best_metrics)
        print(f"Fold {fold+1} Best Val Acc: {fold_best_val_acc:.2f}%")

    ######################################################
    #_________________Final Test Dataset_________________#
    ######################################################
    print("\nEvaluating Best Overall Model on Test Set...")
    model.load_state_dict(torch.load(best_overall_model_path))
    model.eval()
    
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device).to(torch.float32)
            # Use the PCA model from the best performing fold
            features = apply_pca(features, pca_model=best_pca_model, train_data=False)
            labels = (batch['label']-1).long().to(device)
            cluster_map = batch['cluster_map'].to(device).to(torch.long)

            outputs = model(features, cluster_map)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, predicted = torch.max(outputs.data, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    test_acc = accuracy_score(y_true, y_pred)
    test_sens = tp/(tp+fn) if (tp+fn) > 0 else 0
    test_spec = tn/(tn+fp) if (tn+fp) > 0 else 0
    test_auroc = roc_auc_score(y_true, y_prob)

    print(f"Test Results -> Acc: {test_acc:.2f}, Sens: {test_sens:.2f}, Spec: {test_spec:.2f}, AUC: {test_auroc:.2f}")

    # Data Handling for Export
    metrics_df = pd.DataFrame(metrics_records)
    best_metrics_df = pd.DataFrame(best_metrics_per_fold)
    
    csv_path = os.path.join(config['output_dir'], "all_epochs_metrics.csv")
    metrics_df.to_csv(csv_path, index=False)
    
    # Calculate Cross-Validation Stats using only the best from each fold
    cv_acc_mean = best_metrics_df['val_acc'].mean()
    cv_acc_std = best_metrics_df['val_acc'].std()
    
    print(f"\nFinal CV Results (Best Epoch per Fold):")
    print(f"Val Accuracy: {cv_acc_mean:.2f} ± {cv_acc_std:.2f}")
    print(f"Val AUROC: {100*best_metrics_df['val_AUROC'].mean():.2f} ± {best_metrics_df['val_AUROC'].std():.2f}")
    
    plot_training_curves(metrics_df, save_dir=config['output_dir'])