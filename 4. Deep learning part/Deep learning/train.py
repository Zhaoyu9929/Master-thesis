import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import LSTMWithMLP
from utils import mdn_loss,EarlyStopping
from data_preprocessing import prepare_lstm_data, prepare_mlp_data
from config import batch_size, epochs, learning_rate, num_gaussians, patience, delta
import torch
from visualization import plot_training_loss

def train_model():
    # Load the data
    windows, _ = prepare_lstm_data()
    mlp_data = prepare_mlp_data()

    # Initialize the model, the loss function and optimizer
    model = LSTMWithMLP(lstm_input_size=1, output_size=1, num_gaussians=num_gaussians, mlp_input_dim=mlp_data.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=patience, verbose=True, delta=delta)

    for window_idx, (X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor) in enumerate(windows):
        if window_idx > 0: 
            # Load the model state from previous window
            model.load_state_dict(torch.load('model_state.pth'))

        # Create the dataset and dataloader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False) 
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 

        train_losses = []
        val_losses = []

        # Set the number of epochs
        for epoch in range(epochs):
            # Calculate the train loss
            model.train()
            train_loss = 0.0

            for batch_idx, (seq, label) in enumerate(train_loader):
                # Clear old gradients 
                optimizer.zero_grad()

                # Prepare MLP input data for the batch
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + seq.size(0), len(mlp_data)) # Use seq.size(0) to handle the last batch which may be smaller
                mlp_batch_input = mlp_data.iloc[start_idx:end_idx].values.astype(float)
                mlp_batch_input = torch.tensor(mlp_batch_input, dtype=torch.float32)

                pi, sigma, mu = model(seq, mlp_batch_input)
                loss = mdn_loss(pi, sigma, mu, label)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * seq.size(0)  # Acutual size of the previous batch

            train_losses.append(train_loss / len(X_train_tensor))

            # Calculate the train loss
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

            # Early stopping check

            # Get the average val loss of all batch in this epoch
            early_stopping(val_loss / len(X_test_tensor))
            if early_stopping.early_stop:
                print("Stopping early due to increasing validation loss.")
                break
        plot_training_loss(train_losses, val_losses, window_idx)
        # Save the current state of this window for use in next window
        torch.save(model.state_dict(), 'model_state.pth')
    
    return model