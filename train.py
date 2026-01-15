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
    # Get the config file
    config = train_config.config
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    os.makedirs(config['output_dir'],exist_ok=True)
    exclude_list=[]
    dataset_len=0
    for i in os.listdir(config['feature_data_dir']): #Iterating through feature matrix to find the subjects without proper shapes and final dataset length after excluding them
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
    
    metrics_records=[]
    for fold,(train_loader,val_loader) in enumerate(folds):
        print(f"Training for fold: {fold+1}")
        print(".............................")
        early_stoper=EarlyStopping(patience=10)
        pca_model=PCA(n_components=config['n_components'])

        #1.Fitting PCA model on Train Set
        all_train_features=[]
        for batch in train_loader:
            features=batch['features'].to(device)
            all_train_features.append(features)
        all_train_features=torch.cat(all_train_features,dim=0)
        B,T,F=all_train_features.shape
        all_train_features=all_train_features.view(B*T,F)
        all_train_features=all_train_features.to(torch.float32)
        pca_model.fit(all_train_features)

        train_losses=[]
        train_accs=[]
        val_losses=[]
        val_accs=[]

        model=AtlasFreeBrainTransformer().to(device)
        loss_func=nn.CrossEntropyLoss()
        optimizer=optim.Adam(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
        best_val_acc=0.0
        Accuracy=0.0
        ###############################################
        #_____________Train On Train Set______________#
        ###############################################
        for epoch in range(config['Epochs']):
            model.train()

            correct=0
            total=0
            running_loss=0.0
            loop=tqdm(train_loader, desc=f"Epoch [{epoch+1}/{config['Epochs']}]")

            for batch in loop:
                features=batch['features'].to(device)
                features=features.to(torch.float32)
                features=apply_pca(features,pca_model=pca_model,train_data=True) #Pca Application on Feature matrix to reduce dimensionality ([Batch, 400, 1632] -> [Batch, 400, 512])

                labels=(batch['label']-1).long().to(device)
                # labels=labels.to(device) #Shape ([Batch, ])

                cluster_map=batch['cluster_map'].to(device)
                cluster_map=cluster_map.to(torch.long) #shape ([Batch, 45, 54, 45])

                outputs=model(features, cluster_map) #Model Output

                loss=loss_func(outputs,labels)
                _,predicted=torch.max(outputs.data,1)
                total+=labels.size(0)
                correct+=(predicted==labels).sum().item()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0) # Gradient Norm clipping to prevent exploding gradients
                optimizer.step()
                running_loss+=loss.item()
                loop.set_postfix(loss=loss.item())

            print(f"Epoch {epoch+1} Complete. Average Loss: {running_loss/len(train_loader):.4f}")
            print(f"Training Accuracy: {100*correct / total:.2f}%")
            train_loss=running_loss/len(train_loader)
            train_accs=100*correct/total

            ######################################################
            #______________Test On Val Set_______________________#
            ######################################################
            y_true=[]
            y_pred=[]
            y_prob=[]
            model.eval()
            with torch.no_grad():
                correct=0
                total=0
                running_loss=0.0
                for batch in val_loader:
                    features=batch['features'].to(device)
                    features=features.to(torch.float32)
                    features=apply_pca(features,pca_model=pca_model,train_data=False) #

                    labels=(batch['label']-1).long().to(device)

                    cluster_map=batch['cluster_map'].to(device)
                    cluster_map=cluster_map.to(torch.long)

                    # labels=labels.to(device)
                    outputs=model(features,cluster_map)

                    probs=torch.softmax(outputs,dim=1)[:,1]

                    loss=loss_func(outputs,labels)
                    _,predicted=torch.max(outputs.data, 1)
                    running_loss+=loss.item()
                    total+=labels.size(0)
                    correct+=(predicted==labels).sum().item()

                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(predicted.cpu().numpy())
                    y_prob.extend(probs.cpu().numpy())
                tn,fp,fn,tp=confusion_matrix(y_true,y_pred,labels=[0,1]).ravel()
                #metrics Calculation
                acc=accuracy_score(y_true,y_pred)
                sens=tp/(tp+fn) if (tp+fn)>0 else 0
                spec=tn/(tn+fp) if (tn+fp)>0 else 0
                auroc= roc_auc_score(y_true,y_prob)

                print(f"Validation Accuracy: {100*correct / total:.2f}%")
                Accuracy+=100*correct/total
                val_accs=100*correct/total
                improved=early_stoper.step(val_accs)
                if improved:
                    best_val_acc=val_accs
                    torch.save(model.state_dict(),f"{config['output_dir']}/best_fold_{fold}.pt")
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
                        "val_acc": val_accs,
                        "val_sensitivity": sens,
                        "val_specificity" : spec,  
                        "val_AUROC" : auroc 
                    })
        print(f"Final Accuracy: {Accuracy/config['Epochs']}%")
        
        print(f"Finish Training for Fold: {fold+1}")
        print("...................................")

    ######################################################
    #_________________Test Dataset_______________________#
    ######################################################
    y_true=[]
    y_pred=[]
    y_prob=[]
    model.eval()
    with torch.no_grad():
        correct=0
        total=0
        for batch in test_loader:
            features=batch['features'].to(device)
            features=features.to(torch.float32)
            features=apply_pca(features,pca_model=pca_model,train_data=False)

            labels=(batch['label']-1).long().to(device)

            cluster_map=batch['cluster_map'].to(device)
            cluster_map=cluster_map.to(torch.long)

            outputs=model(features, cluster_map)
            probs=torch.softmax(outputs,dim=1)[:,1]
            _,predicted=torch.max(outputs.data,1)
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())
        tn,fp,fn,tp=confusion_matrix(y_true,y_pred,labels=[0,1]).ravel()
        #metrics Calculation for test set
        acc=accuracy_score(y_true,y_pred)
        sens=tp/(tp+fn) if (tp+fn)>0 else 0
        spec=tn/(tn+fp) if (tn+fp)>0 else 0
        auroc= roc_auc_score(y_true,y_prob)
        print(f"Accuracy: {acc}, Sensitivity: {sens}, Specificity: {spec}, AUROC: {auroc}")
        metrics_records.append({
            "test_acc" : acc,
            "test_sensitivity" : sens,
            "test_specificity": spec,
            "test_AUROC": auroc
        })
    
    metrics_df=pd.DataFrame(metrics_records)
    csv_path=os.path.join(config['output_dir'],"metrics.csv")
    metrics_df.to_csv(csv_path,index=False)
    print(f'Val_Accuracy: {metrics_df['val_acc'].mean()}±{metrics_df['val_acc'].std()}|Val_sensitivity: {metrics_df['val_sensitivity'].mean()}±{metrics_df['val_sensitivity'].std()}| Val_Specificity: {metrics_df['val_specificity'].mean()}±{metrics_df['val_specificity'].std()} |  Val_AUROC : {metrics_df['val_AUROC'].mean()}± {metrics_df['val_AUROC'].std()}')

    print(f"Saved_metrics to {csv_path}")
    
    plot_training_curves(metrics_df,save_dir=config['output_dir'])
    

