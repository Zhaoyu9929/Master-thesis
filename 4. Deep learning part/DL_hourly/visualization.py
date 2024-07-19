import matplotlib.pyplot as plt

# Visualize the training loss
def plot_training_loss(train_losses, val_losses, window_idx):
    epochs = range(len(train_losses))
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title(f'Training and Validation Loss for Window {window_idx}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_nll_loss(nll_losses):
    
    plt.figure(figsize=(10, 5))
    plt.plot(nll_losses, label='Negative Log-Likelihood')
    plt.xlabel('Test Sample Index')
    plt.ylabel('Negative Log-Likelihood')
    plt.title('Negative Log-Likelihood Across Test Samples')
    plt.legend()
    plt.show()
