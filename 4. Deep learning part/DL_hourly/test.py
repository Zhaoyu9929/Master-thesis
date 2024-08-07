import numpy as np
import torch
from utils import mdn_loss_test
from data_preprocessing import prepare_lstm_data, prepare_mlp_data, inverse_normalize_count

def test_model(model):
    model.eval()
    windows, scaler = prepare_lstm_data()
    mlp_data = prepare_mlp_data()

    total_nll_losses = []
    total_mae_losses = []
    total_mape_losses = []
    total_r2_scores = []
    total_picp = []
    total_ll = []

    for _, _, X_test_tensor, y_test_tensor in windows:
        
        nll_losses = []
        mae_losses = []
        mape_losses = []
        all_predicted = []
        all_actual = []
        picp_coverage = []
        log_likelihoods = []

    
        with torch.no_grad():
            for i in range(len(X_test_tensor)):
                seq = X_test_tensor[i:i+1]
                label = y_test_tensor[i:i+1]

                # Prepare MLP input data for the test
                mlp_input = mlp_data.iloc[i:i+1].values.astype(float)
                mlp_input = torch.tensor(mlp_input, dtype=torch.float32)

                # Get the model output
                pi, sigma, mu = model(seq, mlp_input)
                loss = mdn_loss_test(pi, sigma, mu, label)
                nll_losses.append(loss.item())

                # Select the Gaussian with the highest probability (pi)
                max_pi_idx = torch.argmax(pi, dim=1)  # dim=1 for the row axis
                predicted_mean_normalized = mu[torch.arange(len(mu)), max_pi_idx]
                predicted_mean = inverse_normalize_count(predicted_mean_normalized, scaler)
                predicted_mean = torch.tensor(predicted_mean, dtype=torch.float32)
                label = inverse_normalize_count(label, scaler)
                label = torch.tensor(label, dtype=torch.float32)

                # Append to all_predicted and all_actual for R^2 calculation
                all_predicted.append(predicted_mean.item())
                all_actual.append(label.item())

                # Calculate MAE 
                mae_loss = torch.mean(torch.abs(predicted_mean - label)).item()
                mae_losses.append(mae_loss)

                # Calculate MAPE, avoid division by zero
                label_np = label.numpy()
                predicted_mean_np = predicted_mean.numpy()
                non_zero_labels = label_np != 0
                if np.any(non_zero_labels):
                    mape_loss = np.mean(np.abs((label_np[non_zero_labels] - predicted_mean_np[non_zero_labels]) / label_np[non_zero_labels])) * 100
                else:
                    mape_loss = float('inf')
                mape_losses.append(mape_loss)
                
                # Calculate PICP
                num_samples = 100
                samples = []
                for _ in range(num_samples):
                    idx = torch.multinomial(pi, 1).item()
                    sampled_value = np.random.normal(mu[0, idx].item(), sigma[0, idx].item())
                    samples.append(sampled_value)

                samples = np.array(samples)
                samples_tensor = torch.tensor(samples, dtype=torch.float32)
                samples_tensor = inverse_normalize_count(samples_tensor, scaler)

                # samples_tensor already is numpy array after inverse_normalize_count
                lower_bound = np.percentile(samples_tensor, 2.5)
                upper_bound = np.percentile(samples_tensor, 97.5)
                coverage = (label.item() >= lower_bound) and (label.item() <= upper_bound)
                coverage = coverage * 100
                picp_coverage.append(coverage)

                # Calculate log-likelihood (LL)
                mixture_prob = 0
                for j in range(len(pi[0])):
                    norm_prob = (1 / np.sqrt(2 * np.pi * sigma[0, j].item()**2)) * np.exp(-0.5 * ((label.item() - mu[0, j].item())**2) / sigma[0, j].item()**2)
                    mixture_prob += pi[0, j].item() * norm_prob

                # Avoid log(0) by adding a small value to mixture_prob
                mixture_prob = max(mixture_prob, 1e-10)
                log_likelihood = np.log(mixture_prob)
                log_likelihoods.append(log_likelihood)


        # Calculate R^2
        all_predicted = np.array(all_predicted)
        all_actual = np.array(all_actual)
        ss_res = np.sum((all_actual - all_predicted) ** 2)
        ss_tot = np.sum((all_actual - np.mean(all_actual)) ** 2)
        r2_score = (1 - (ss_res / ss_tot)) 
                
        total_nll_losses.append(np.mean(nll_losses))
        total_mae_losses.append(np.mean(mae_losses))
        total_mape_losses.append(np.mean(mape_losses))
        total_picp.append(np.mean(picp_coverage))
        total_ll.append(np.mean(log_likelihoods))
        total_r2_scores.append(r2_score)

    # Calulate average of metrics across all windows
    average_nll_loss = np.mean(nll_losses)
    average_mae_loss = np.mean(mae_losses)
    average_mape_loss = np.mean(mape_losses)
    average_picp = np.mean(total_picp)
    average_ll = np.mean(total_ll)
    average_r2_score = np.mean(total_r2_scores)



    return average_nll_loss, average_mae_loss, average_mape_loss, average_r2_score, average_picp, average_ll