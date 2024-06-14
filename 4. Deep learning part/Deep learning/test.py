import pandas as pd
import torch
from utils import mdn_loss
from data_preprocessing import prepare_lstm_data, prepare_mlp_data

"""
Here each NLL loss represents the negative log-likelihood loss of the model on the test sample
"""
def test_model(model):
    # Calculating negative log-likelihood for the test set
    nll_losses = []
    mse_losses = []
    mae_losses = []
    model.eval()
    _, _, X_test_tensor, y_test_tensor, _ = prepare_lstm_data()
    mlp_data = prepare_mlp_data()


    with torch.no_grad():
        for i in range(len(X_test_tensor)):
            seq = X_test_tensor[i:i+1]
            label = y_test_tensor[i:i+1]

            # Prepare MLP input data for the test
            mlp_input = mlp_data.iloc[i:i+1].values.astype(float)
            mlp_input = torch.tensor(mlp_input, dtype=torch.float32)

            # Get the model output
            pi, sigma, mu = model(seq, mlp_input)
            loss = mdn_loss(pi, sigma, mu, label)
            nll_losses.append(loss.item())

            # Select the Gaussian with the highest probability (pi)
            max_pi_idx = torch.argmax(pi, dim=1)  # dim=1 for the row axis
            predicted_mean = mu[torch.arange(len(mu)), max_pi_idx]

            # Calculate MSE and MAE losses
            mse_loss = torch.mean((predicted_mean - label) ** 2).item()
            mae_loss = torch.mean(torch.abs(predicted_mean - label)).item()

            mse_losses.append(mse_loss)
            mae_losses.append(mae_loss)

    average_nll_loss = sum(nll_losses) / len(nll_losses)
    average_mse_loss = sum(mse_losses) / len(mse_losses)
    average_mae_loss = sum(mae_losses) / len(mae_losses)

    return nll_losses, average_nll_loss, average_mse_loss, average_mae_loss
