# import argparse
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW, Adam
from sklearn.metrics import accuracy_score, classification_report
from torch import nn
from collections import defaultdict
from torch import cuda, nn,optim
from tqdm import tqdm
from datetime import datetime
from AFT import AtlasFreeBrainTransformer
from torch_pca import PCA
from APPLY_PCA import apply_pca
from sklearn.model_selection import KFold
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
    # Get the config file
    config = train_config.config
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    
    print("PCA Model Loaded!")
    print("Selected Device:", device)
    exclude_list=[]
    dataset_len=0
    for i in os.listdir(config['feature_data_dir']):
        data=np.load(os.path.join(config['feature_data_dir'],i),allow_pickle=True)
        data=data.item()['feature_mat']
        dataset_len+=1
        if data.shape[0]!=400:
            exclude_list.append(int(i.split('_')[1]))
            dataset_len-=1
    print(f"Subjects without proper shape: {exclude_list}")
    folds, test_loader = create_dataloaders(
                                            label_data_dir=config['label_data_dir'],
                                            feature_data_dir=config['feature_data_dir'],
                                            cluster_data_dir=config['cluster_data_dir'],
                                            batch_size=config['batch_size'],
                                            exclude_list=exclude_list
                                        )
    indices=np.arange(dataset_len)
    k_fold=KFold(n_splits=config['n_split'],shuffle=True,random_state=42)
    
    metrics_records=[]
    for fold,(train_loader,val_loader) in enumerate(folds):
        print(f"Training for fold: {fold+1}")
        print(".............................")
        early_stoper=EarlyStopping(patience=10)
        pca_model=PCA(n_components=config['n_components'])
        all_train_features=[]
        for batch in train_loader:
            features=batch['features'].to(device)
            all_train_features.append(features)
        all_train_features=torch.cat(all_train_features,dim=0)
        B,T,F=all_train_features.shape
        all_train_features=all_train_features.view(B*T,F)
        # print(all_train_features.shape)
        all_train_features=all_train_features.to(torch.float32)
        pca_model.fit(all_train_features)

        train_losses=[]
        train_accs=[]
        val_losses=[]
        val_accs=[]
        epochs_list=[]

        model=AtlasFreeBrainTransformer().to(device)
        loss_func=nn.CrossEntropyLoss()
        optimizer=optim.Adam(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
        # scheduler=ReduceLROnPlateau(optimizer, mode='max',factor=0.01, patience=6, min_lr=2e-4)
        best_val_acc=0.0

        Accuracy=0.0
        for epoch in range(config['Epochs']):
            ###############################################
            #_______________Training Dataset______________#
            ###############################################
            model.train()
            correct=0
            total=0
            running_loss=0.0
            loop=tqdm(train_loader, desc=f"Epoch [{epoch+1}/{config['Epochs']}]")
            for batch in loop:
                features=batch['features'].to(device)
                features=features.to(torch.float32)
                features=apply_pca(features,pca_model=pca_model,train_data=True) #Apply PCA to reduce dimensionality
                labels=(batch['label']-1).long().to(device)
                labels=labels.to(device)
                cluster_map=batch['cluster_map'].to(device)
                cluster_map=cluster_map.to(torch.long)
                outputs=model(features, cluster_map) #Get prediction from the model
                loss=loss_func(outputs,labels)
                _,predicted=torch.max(outputs.data,1)
                total+=labels.size(0)
                correct+=(predicted==labels).sum().item()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
                optimizer.step()
                running_loss+=loss.item()
                loop.set_postfix(loss=loss.item())
            print(f"Epoch {epoch+1} Complete. Average Loss: {running_loss/len(train_loader):.4f}")
            print(f"Training Accuracy: {100*correct / total:.2f}%")
            train_loss=running_loss/len(train_loader)
            train_accs=100*correct/total
            # epochs_list.append(epoch)

            ######################################################
            #_____________________Validation Dataset_____________#
            ######################################################

            model.eval()
            with torch.no_grad():
                correct=0
                total=0
                running_loss=0.0
                for batch in val_loader:
                    features=batch['features'].to(device)
                    features=features.to(torch.float32)
                    features=apply_pca(features,pca_model=pca_model,train_data=False)
                    labels=batch['label']-1
                    cluster_map=batch['cluster_map'].to(device)
                    cluster_map=cluster_map.to(torch.long)
                    labels=labels.to(device)
                    outputs=model(features,cluster_map)
                    loss=loss_func(outputs,labels)
                    _,predicted=torch.max(outputs.data, 1)
                    running_loss+=loss.item()
                    total+=labels.size(0)
                    correct+=(predicted==labels).sum().item()
                print(f"Validation Accuracy: {100*correct / total:.2f}%")
                Accuracy+=100*correct/total
                val_accs=100*correct/total
                improved=early_stoper.step(val_accs)
                if improved:
                    best_val_acc=val_accs
                    torch.save(model.state_dict(),f"best_fold_{fold}.pt")
                if early_stoper.stop:
                    print("Early_stoppint.")
                    break
                val_losses=running_loss/len(val_loader)
                metrics_records.append({
                        "fold": fold+1,
                        "epoch": epoch+1,
                        "train_loss": train_loss,
                        "train_acc": train_accs,
                        "val_loss": val_losses,
                        "val_acc": val_accs
                    })
        print(f"Final Accuracy: {Accuracy/config['Epochs']}%")
        
        print(f"Finish Training for Fold: {fold+1}")
        print("...................................")

    metrics_df=pd.DataFrame(metrics_records)
    os.makedirs(config['output_dir'],exist_ok=True)
    csv_path=os.path.join(config['output_dir'],"metrics.csv")
    metrics_df.to_csv(csv_path,index=False)
    print(f"Saved_metrics to {csv_path}")
    ######################################################
    #_________________Test Dataset_______________________#
    ######################################################
    model.eval()
    with torch.no_grad():
        correct=0
        total=0
        for batch in test_loader:
            features=batch['features'].to(device)
            features=features.to(torch.float32)
            features=apply_pca(features,pca_model=pca_model,train_data=False)
            labels=batch['label']-1
            cluster_map=batch['cluster_map'].to(device)
            cluster_map=cluster_map.to(torch.long)
            labels=labels.to(device)
            outputs=model(features, cluster_map)
            _,predicted=torch.max(outputs.data,1)
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
        print(f"Test Accuracy: {100*correct/ total:.2f}%")
    print(train_losses, train_accs)
    
    plot_training_curves(metrics_df,save_dir=config['output_dir'])
    


        

    





    
    # Get one batch to verify
    # batch = next(iter(train_loader))
    # print(batch.keys())
    # print("\nBatch shapes:")
    # print(f"features    : {batch['features'].shape}")     # [batch, 400, 1632]
    # print(f"labels      : {batch['label'].shape}") 
    # print(f"cluster     : {batch['cluster_map'].shape}")       # [batch]
    # print(f"subject_ids : {batch['subject_id']}")
    # print(f"feature dtype: {batch['features'].dtype}")
    # print(f"cluster dtype: {batch['cluster_map'].dtype}")

    # if 'cluster_map' in batch:
    #     print(f"cluster_map : {batch['cluster_map'].shape}")

    # F=apply_pca(batch['features'])    
    # model = AtlasFreeBrainTransformer()
    
    # # F = torch.randn(2, 400, 1632)
    # # C = torch.randint(0, 401, (2, 45, 54, 45))
    # batch['cluster_map']=batch['cluster_map'].to(torch.long)
    # logits = model(F, batch['cluster_map'])
    # print(logits)  # (2, 2)
    

