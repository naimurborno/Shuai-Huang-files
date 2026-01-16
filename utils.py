import matplotlib.pyplot as plt
import os

class EarlyStopping:
    def __init__(self,patience=5):
        self.patience=patience
        self.best=None
        self.counter=0
        self.stop=False
    def step(self,score):
        if self.best is None or score>self.best:
            self.best=score
            self.counter=0
            return True
        else:
            self.counter+=1
            if self.counter>=self.patience:
                self.stop=True
            return False
        
def plot_training_curves(metrics_df, save_dir="outputs"):
    """
    Plots validation accuracy vs epoch for each fold.
    Saves the figure as 'val_accuracy_per_fold.png'.
    """

    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))

    #Training Accuracy
    for fold in metrics_df["fold"].unique():
        fold_df = metrics_df[metrics_df["fold"] == fold]
        plt.plot(
            fold_df["epoch"],
            fold_df["train_acc"],
            label=f"Fold {fold}"
        )
    plt.xlabel("Epoch")
    plt.ylabel("Training Accuracy (%)")
    plt.title("Training Accuracy per Fold")
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(save_dir, "train_accuracy_per_fold.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    #Training Loss
    for fold in metrics_df["fold"].unique():
        fold_df=metrics_df[metrics_df['fold']==fold]
        plt.plot(
            fold_df['epoch'],
            fold_df['train_loss'],
            label=f"Fold{fold}"
        )
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Train Loss per Fold")
    plt.legend()
    plt.grid(True)
    save_path=os.path.join(save_dir,"train_loss_per_fold.png")
    plt.savefig(save_path,dpi=300,bbox_inches="tight")
    plt.close()

    print(f"Training curves saved to: {save_path}")

    #ROC Curve

    for fold in metrics_df["fold"].unique():
        fold_df=metrics_df[metrics_df['fold']==fold]
        plt.plot(
            fold_df['epoch'],
            fold_df['val_AUROC'],
            label=f"Fold{fold}"
        )
    plt.xlabel("Epoch")
    plt.ylabel("Validation AUROC")
    plt.title("Validation AUROC per Fold")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(save_dir, "val_auroc_per_fold.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Training curves saved to: {save_path}")