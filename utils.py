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

    for fold in metrics_df["fold"].unique():
        fold_df = metrics_df[metrics_df["fold"] == fold]
        plt.plot(
            fold_df["epoch"],
            fold_df["val_acc"],
            label=f"Fold {fold}"
        )

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Validation Accuracy per Fold")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(save_dir, "val_accuracy_per_fold.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    for fold in metrics_df["fold"].unique():
        fold_df=metrics_df[metrics_df['fold']==fold]
        plt.plot(
            fold_df['epoch'],
            fold_df['val_loss'],
            label=f"Fold{fold}"
        )
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss per Fold")
    plt.legend()
    plt.grid(True)
    save_path=os.path.join(save_dir,"val_loss_per_fold.png")
    plt.savefig(save_path,dpi=300,bbox_inces="tight")
    plt.close()

    print(f"Training curves saved to: {save_path}")


    # Loss plot
    # ax1.plot(epochs, train_loss, 'b-', label='Train Loss')
    # ax1.plot(epochs, val_loss, 'r-', label='Validation Loss')
    # ax1.set_title('Loss over Epochs')
    # ax1.set_xlabel('Epoch')
    # ax1.set_ylabel('Loss')
    # ax1.legend()
    # ax1.grid(True, alpha=0.3)
    
    # # Accuracy plot
    # # print(epochs)
    # # print(train_acc.shape)
    # # print(val_acc.shape)
    # # print(epochs)
    # # print(train_acc)
    # # print(val_acc)
    # ax2.plot(epochs, train_acc, 'b-', label='Train Accuracy')
    # ax2.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    # ax2.set_title('Accuracy over Epochs')
    # ax2.set_xlabel('Epoch')
    # ax2.set_ylabel('Accuracy (%)')
    # ax2.set_ylim(0, 100)
    # ax2.legend()
    # ax2.grid(True, alpha=0.3)
    
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_path,"training figure.jpg"), dpi=300, bbox_inches='tight')
    # plt.close(fig)  # close figure to free memory
    # print(f"Training curves saved to: {save_path}")