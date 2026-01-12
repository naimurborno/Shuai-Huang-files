import argparse
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, classification_report
from torch import nn
from collections import defaultdict
from torch import cuda, nn,optim
from tqdm import tqdm
from datetime import datetime
from AFT import AtlasFreeBrainTransformer
from torch_pca import PCA
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
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") #Check if cuda is available
    print("Selected Device:", device)
    pca_model=PCA(n_components=config['n_components'])
    train_loader, val_loader, test_loader = create_dataloaders(
                                            label_data_dir=config['label_data_dir'],
                                            feature_data_dir=config['feature_data_dir'],
                                            cluster_data_dir=config['cluster_data_dir'],
                                            batch_size=config['batch_size']
                                        )
    all_train_features=[]
    for batch in train_loader:
        features=batch['features'].to(device)
        B, T, F=features.shape
        features=features.view(B*T,F)
        all_train_features.append(features)
    pca_model.fit(all_train_features)

    

    model=AtlasFreeBrainTransformer().to(device)
    loss_func=nn.BCELoss()
    optimizer=optim.AdamW(model.parameters(),lr=1e-6)
    for epoch in range(config['Epochs']):
        model.train()
        running_loss=0.0
        loop=tqdm(train_loader, desc=f"Epoch [{epoch+1}/{config['Epochs']}]")
        for batch in loop:
            features=batch['features'].to(device)
            # features=features.to(torch.float32)
            labels=batch['label']-1
            labels=labels.to(device)
            cluster_map=batch['cluster_map'].to(device)
            cluster_map=cluster_map.to(torch.long)
            features=apply_pca(features,pca_model=pca_model,train_data=True) #Apply PCA to reduce dimensionality
            # print("Data size after applying PCA:", features.shape)
            outputs=model(features, cluster_map) #Get prediction from the model
            # _, predicted=torch.max(outputs, dim=1)
            # print("shape of output:", outputs.shape)
            # print("outputs:",outputs)
            # print("labels:",labels)
            # print("shape of labels:", labels.shape)


            loss=loss_func(outputs,labels.float().view_as(outputs))
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            running_loss+=loss.item()
            loop.set_postfix(loss=loss.item())
        print(f"Epoch {epoch+1} Complete. Average Loss: {running_loss/len(train_loader):.4f}")

        model.eval()
        with torch.no_grad():
            correct=0
            total=0
            for batch in val_loader:
                features=batch['features'].to(device)
                # features=features.to(torch.float32)
                features=apply_pca(features,pca_model=pca_model,train_data=False)
                labels=batch['label']-1
                cluster_map=batch['cluster_map'].to(device)
                cluster_map=cluster_map.to(torch.long)
                labels=labels.to(device)
                outputs=model(features,cluster_map)
                # _,predicted=torch.max(outputs.data, 1)
                predicted=(outputs>0.5).float()
                predicted=predicted.view(-1)
                labels=labels.view(-1)
                print("This is predicted: ",predicted)
                print("This is original label: ",labels)
                total+=labels.size(0)
                correct+=(outputs==labels).sum().item()
            print(f"Validation Accuracy: {100*correct / total:.2f}%")
        

    





    
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
    

