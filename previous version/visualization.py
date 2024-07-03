import matplotlib.pyplot as plt
from train import train_model

# Visualize the training loss
def plot_training_loss(train_losses, val_losses):
    epochs = range(len(train_losses))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_nll_loss(nll_losses):
    
    plt.figure(figsize=(10, 5))
    plt.plot(nll_losses, label='Negative Log-Likelihood')
    plt.xlabel('Test Sample Index')
    plt.ylabel('Negative Log-Likelihood')
    plt.title('Negative Log-Likelihood Across Test Samples')
    plt.legend()
    plt.show()