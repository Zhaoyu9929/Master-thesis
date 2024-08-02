from train import train_model
from test import test_model
import torch
from model import LSTMWithMLP
import optuna

def main():
    # Train the model
    model = train_model()

    # model = LSTMWithMLP(lstm_input_size=1, output_size=1, num_gaussians=5, mlp_input_dim=11)
    # model.load_state_dict(torch.load('final_model_state.pth'))

    # Test the model
    average_nll_loss, average_mae_loss, average_mape_loss, average_r2_score, average_picp, average_ll = test_model(model)

    # # Output the average losses and R^2 score
    print(f'Average NLL Loss on Test Set: {average_nll_loss:.4f}')
    print(f'Average MAE on Test Set: {average_mae_loss:.4f}')
    print(f'Average MAPE on Test Set: {average_mape_loss:.2f}')
    print(f'R^2 Score on Test Set: {average_r2_score:.4f}')
    print(f'Average PICP on Test Set: {average_picp:.2f}')
    print(f'Average Log-Likelihood on Test Set: {average_ll:.4f}')

    # Save the model state
    torch.save(model.state_dict(), 'final_model_state.pth')
    print("Model state saved successfully.")

if __name__ == "__main__":
    main()
