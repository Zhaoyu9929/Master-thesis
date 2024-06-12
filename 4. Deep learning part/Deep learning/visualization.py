import matplotlib.pyplot as plt
from train import train_model

# Visualize the training loss
def plot_training_loss(train_losses):

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training loss')
    plt.xlabel('Epoch')
    plt.ylabel('Training loss')
    plt.title('Training loss over epoch')
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
