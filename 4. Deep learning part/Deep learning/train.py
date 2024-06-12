import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import LSTMMDNModel
from utils import mdn_loss
from data_preprocessing import prepare_lstm_data
from config import batch_size, epochs, learning_rate, seq_length
import torch

def train_model(lstm_data, mlp_data):
    X_train_tensor, y_train_tensor, _, _ = prepare_lstm_data(lstm_data, seq_length)
    
    # Create the dataset and dataloader
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    # Randomly shuffle the data at the beginning of each training cycle (epoch), avoiding overfitting during training
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) 

    # Initialize the model, the loss function and optimizer
    model = LSTMMDNModel()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_losses = []

    # Set the number of epochs
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for i in range(len(data_loader)):
            seq, label = data_loader.dataset[i]
            # Clear old gradients
            optimizer.zero_grad()

            mlp_input = torch.tensor(mlp_data.iloc[i:i+batch_size].values, dtype=torch.float32)
            pi, sigma, mu = model(seq, mlp_input)

            loss = mdn_loss(pi, sigma, mu, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_size 

        train_losses.append(train_loss / len(X_train_tensor))

        if epoch % 10 == 0:
            print(f'Epoch {epoch} Train Loss: {train_losses[-1]}')

    print(f'Final Train Loss: {train_losses[-1]}')
    return model