from test import model
import torch
from utils import mdn_loss
from data_preprocessing import prepare_lstm_data

def test_model():
    # Calculating negative log-likelihood for the test set
    nll_losses = []
    model.eval()
    _, _, X_test_tensor, y_test_tensor = prepare_lstm_data()
    with torch.no_grad():
        for i in range(len(X_test_tensor)):
            seq = X_test_tensor[i:i+1]
            label = y_test_tensor[i:i+1]

            pi, sigma, mu = model(seq)
            loss = mdn_loss(pi, sigma, mu, label)
            nll_losses.append(loss.item())
    return model


