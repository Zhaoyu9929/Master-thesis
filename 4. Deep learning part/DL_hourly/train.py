import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import LSTMWithMLP
from utils import mdn_loss, EarlyStopping
from data_preprocessing import prepare_lstm_data, prepare_mlp_data
from config import batch_size, epochs, learning_rate, num_gaussians, patience, delta
import torch
from visualization import plot_training_loss
import optuna

def train_model(trial):
    # Hyperparameter search range
    lstm_hidden_layer_size = trial.suggest_int('lstm_hidden_layer_size', 32, 128)
    mlp_hidden_dim = trial.suggest_int('mlp_hidden_dim', 4, 35)
    learning_rate = trial.suggest_categorical('learning_rate', [0.000001, 0.00001, 0.0001, 0.001])
    dropout = trial.suggest_categorical('dropout', [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    num_gaussians = trial.suggest_int('num_gaussians', 1, 10)
    patience = trial.suggest_int('patience', 3, 10)

    # Load the data
    windows, _ = prepare_lstm_data()
    mlp_data = prepare_mlp_data()

    all_val_losses = []
    for window_idx, (X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor) in enumerate(windows):
        # Initialize the model, the loss function, and optimizer for each window
        model = LSTMWithMLP(
            lstm_input_size=1, 
            output_size=1, 
            num_gaussians=num_gaussians, 
            mlp_input_dim=mlp_data.shape[1],
            lstm_hidden_layer_size=lstm_hidden_layer_size,
            mlp_hidden_dim=mlp_hidden_dim,
            dropout=dropout
            )
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
        early_stopping = EarlyStopping(patience=patience, verbose=True, delta=delta)
        
        # Create the dataset and dataloader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False) 
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 

        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0

            for batch_idx, (seq, label) in enumerate(train_loader):
                optimizer.zero_grad()

                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + seq.size(0), len(mlp_data))
                mlp_batch_input = mlp_data.iloc[start_idx:end_idx].values.astype(float)
                mlp_batch_input = torch.tensor(mlp_batch_input, dtype=torch.float32)

                pi, sigma, mu = model(seq, mlp_batch_input)
                loss = mdn_loss(pi, sigma, mu, label)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * seq.size(0)

            train_losses.append(train_loss / len(X_train_tensor))

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_idx, (seq, label) in enumerate(test_loader):
                    start_idx = batch_idx * batch_size
                    end_idx = start_idx + seq.size(0)
                    mlp_batch_input = mlp_data.iloc[start_idx:end_idx].values.astype(float)
                    mlp_batch_input = torch.tensor(mlp_batch_input, dtype=torch.float32)

                    pi, sigma, mu = model(seq, mlp_batch_input)
                    loss = mdn_loss(pi, sigma, mu, label)
                    val_loss += loss.item() * seq.size(0)
    
            val_losses.append(val_loss / len(X_test_tensor))

            if epoch % 5 == 0:
                print(f'Window {window_idx} Epoch {epoch} Train Loss: {train_losses[-1]}, Val Loss: {val_losses[-1]}')

            early_stopping(val_loss / len(X_test_tensor))
            if early_stopping.early_stop:
                print("Stopping early due to increasing validation loss.")
                break

        # plot_training_loss(train_losses, val_losses, window_idx)
        # model_state_path = r"C:\Users\yanzh\Desktop\code_and_data\4. Deep learning part\处理数据\model_state.pth"
        # torch.save(model.state_dict(), model_state_path)

        all_val_losses.append(val_loss)
        average_val_loss = sum(val_losses) / len(val_losses)
    return average_val_loss


