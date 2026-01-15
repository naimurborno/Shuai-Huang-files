import numpy as np
import torch
from torch import nn, optim
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from collections import defaultdict, Counter
from tqdm import tqdm
import os
import pandas as pd
import warnings

# Custom imports from your provided files
from AFT import AtlasFreeBrainTransformer
from torch_pca import PCA
from APPLY_PCA import apply_pca
from utils import plot_training_curves, EarlyStopping
import train_config
from dataloader import create_dataloaders

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    config = train_config.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    os.makedirs(config['output_dir'], exist_ok=True)
    
    exclude_list = []
    for i in os.listdir(config['feature_data_dir']):
        data = np.load(os.path.join(config['feature_data_dir'], i), allow_pickle=True)
        data = data.item()['feature_mat']
        if data.shape[0] != 400:
            exclude_list.append(int(i.split('_')[1]))
    
    folds, test_loader, full_loader = create_dataloaders(
                                            label_data_dir=config['label_data_dir'],
                                            feature_data_dir=config['feature_data_dir'],
                                            cluster_data_dir=config['cluster_data_dir'],
                                            batch_size=config['batch_size'],
                                            exclude_list=exclude_list
                                        )
    
    metrics_records = []
    num_reps = 10
    best_overall_model_path = os.path.join(config['output_dir'], "best_overall_model.pt")
    absolute_best_val_acc = 0.0

    for fold, (train_loader, val_loader) in enumerate(folds):
        print(f"\nTRAINING FOLD: {fold+1}")
        
        # Repetition buffers for Majority Vote logic
        fold_rep_preds = defaultdict(list)
        fold_rep_probs = defaultdict(list)
        fold_rep_losses = defaultdict(list)
        subject_true_labels = {}

        for rep in range(num_reps):
            print(f"Repetition {rep+1}/{num_reps}")
            early_stoper = EarlyStopping(patience=10)
            pca_model = PCA(n_components=config['n_components'])

            # Fit PCA on Train Set
            all_train_features = []
            for batch in train_loader:
                all_train_features.append(batch['features'].to(device))
            all_train_features = torch.cat(all_train_features, dim=0)
            pca_model.fit(all_train_features.view(-1, all_train_features.shape[-1]).to(torch.float32))

            model = AtlasFreeBrainTransformer().to(device)
            loss_func = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
            
            rep_last_loss = 0.0

            for epoch in range(config['Epochs']):
                model.train()
                running_loss = 0.0
                for batch in train_loader:
                    features = apply_pca(batch['features'].to(device).to(torch.float32), pca_model, True)
                    labels = (batch['label']-1).long().to(device)
                    outputs = model(features, batch['cluster_map'].to(device).to(torch.long))
                    
                    loss = loss_func(outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    running_loss += loss.item()

                rep_last_loss = running_loss / len(train_loader)

                # Validation
                model.eval()
                val_correct, val_total = 0, 0
                with torch.no_grad():
                    for batch in val_loader:
                        features = apply_pca(batch['features'].to(device).to(torch.float32), pca_model, False)
                        labels = (batch['label']-1).long().to(device)
                        outputs = model(features, batch['cluster_map'].to(device).to(torch.long))
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()

                val_acc = 100 * val_correct / val_total
                if early_stoper.step(val_acc):
                    torch.save(model.state_dict(), f"{config['output_dir']}/temp_fold_rep.pt")
                if early_stoper.stop: break

            # Collect Repetition Results
            model.load_state_dict(torch.load(f"{config['output_dir']}/temp_fold_rep.pt"))
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    subj_ids = batch['subject_id']
                    features = apply_pca(batch['features'].to(device).to(torch.float32), pca_model, False)
                    outputs = model(features, batch['cluster_map'].to(device).to(torch.long))
                    probs = torch.softmax(outputs, dim=1)[:, 1]
                    _, predicted = torch.max(outputs, 1)
                    
                    for i, s_id in enumerate(subj_ids):
                        s_id_val = s_id.item()
                        fold_rep_preds[s_id_val].append(predicted[i].item())
                        fold_rep_probs[s_id_val].append(probs[i].item())
                        fold_rep_losses[s_id_val].append(rep_last_loss)
                        subject_true_labels[s_id_val] = (batch['label'][i] - 1).item()

        # --- Majority Vote per Fold ---
        y_true_fold, y_pred_fold, y_prob_fold = [], [], []
        for s_id in fold_rep_preds.keys():
            preds = fold_rep_preds[s_id]
            losses = fold_rep_losses[s_id]
            probs = fold_rep_probs[s_id]
            
            counts = Counter(preds)
            most_common = counts.most_common()
            
            if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
                idx = np.argmin(losses) # Tie-breaker: lowest training loss
                final_label = preds[idx]
                final_prob = probs[idx]
            else:
                final_label = most_common[0][0]
                final_prob = np.mean(probs) # Average probability for AUROC
            
            y_true_fold.append(subject_true_labels[s_id])
            y_pred_fold.append(final_label)
            y_prob_fold.append(final_prob)

        # Fold Metrics
        tn, fp, fn, tp = confusion_matrix(y_true_fold, y_pred_fold, labels=[0,1]).ravel()
        fold_acc = accuracy_score(y_true_fold, y_pred_fold)
        fold_metrics = {
            "fold": fold + 1,
            "val_acc": fold_acc * 100,
            "val_sensitivity": tp/(tp+fn) if (tp+fn)>0 else 0,
            "val_specificity": tn/(tn+fp) if (tn+fp)>0 else 0,
            "val_AUROC": roc_auc_score(y_true_fold, y_prob_fold)
        }
        metrics_records.append(fold_metrics)
        print(f"Fold {fold+1} Voted Acc: {fold_metrics['val_acc']:.2f}%")

        # Save Overall Best Model for Final Test
        if fold_metrics['val_acc'] > absolute_best_val_acc:
            absolute_best_val_acc = fold_metrics['val_acc']
            torch.save(model.state_dict(), best_overall_model_path)

    # --- Final Test Set Evaluation ---
    print("\nEvaluating Final Test Set...")
    model.load_state_dict(torch.load(best_overall_model_path))
    model.eval()
    y_true_test, y_pred_test, y_prob_test = [], [], []
    
    # Using PCA from full_loader (standard practice for final test)
    pca_test = PCA(n_components=config['n_components'])
    all_features = []
    for batch in full_loader: all_features.append(batch['features'])
    all_features = torch.cat(all_features, dim=0)
    pca_test.fit(all_features.view(-1, all_features.shape[-1]).to(torch.float32))

    with torch.no_grad():
        for batch in test_loader:
            features = apply_pca(batch['features'].to(device).to(torch.float32), pca_test, False)
            outputs = model(features, batch['cluster_map'].to(device).to(torch.long))
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, predicted = torch.max(outputs, 1)
            y_true_test.extend((batch['label']-1).numpy())
            y_pred_test.extend(predicted.cpu().numpy())
            y_prob_test.extend(probs.cpu().numpy())

    tn, fp, fn, tp = confusion_matrix(y_true_test, y_pred_test, labels=[0,1]).ravel()
    test_results = {
        "test_acc": accuracy_score(y_true_test, y_pred_test),
        "test_sensitivity": tp/(tp+fn) if (tp+fn)>0 else 0,
        "test_specificity": tn/(tn+fp) if (tn+fp)>0 else 0,
        "test_AUROC": roc_auc_score(y_true_test, y_prob_test)
    }
    
    # Save and Print Results
    metrics_df = pd.DataFrame(metrics_records)
    print(f"\nFinal CV: {metrics_df['val_acc'].mean():.2f}±{metrics_df['val_acc'].std():.2f}")
    print(f"Test Set: Acc: {test_results['test_acc']:.2f}, AUC: {test_results['test_AUROC']:.2f}")
    metrics_df.to_csv(os.path.join(config['output_dir'], "metrics.csv"), index=False)