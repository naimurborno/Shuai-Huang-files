import matplotlib.pyplot as plt
def plot_training_curves(epochs, train_loss, train_acc, val_loss, val_acc, save_path="training_curves.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    # ax1.plot(epochs, train_loss, 'b-', label='Train Loss')
    # ax1.plot(epochs, val_loss, 'r-', label='Validation Loss')
    # ax1.set_title('Loss over Epochs')
    # ax1.set_xlabel('Epoch')
    # ax1.set_ylabel('Loss')
    # ax1.legend()
    # ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, train_acc, 'b-', label='Train Accuracy')
    ax2.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    ax2.set_title('Accuracy over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # close figure to free memory
    print(f"Training curves saved to: {save_path}")