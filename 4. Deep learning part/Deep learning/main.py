from train import train_model
from visualization import plot_training_loss
from test import test_model

def main():

    # Train the model
    model, train_losses, val_losses = train_model()

    # Visualize the training loss and V
    plot_training_loss(train_losses, val_losses)

    # Test the model
    _, average_nll_loss, average_mse_loss, average_mae_loss = test_model(model)
    # Cacluate the average nll_loss
    print(f'Average NLL Loss on Test Set: {average_nll_loss:.4f}')
    print(f'Average MSE on Test Set: {average_mse_loss:.4f}')
    print(f'Average MAE on Test Set: {average_mae_loss:.4f}')
    
    
    


if __name__ == "__main__":
    main()
