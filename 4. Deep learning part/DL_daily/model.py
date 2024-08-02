import torch
import torch.nn as nn
import torch.nn.functional as F
from config import num_gaussians, lstm_hidden_layer_size, dropout, mlp_hidden_dim, l2_lambda, lstm_layer, mdn_hidden_layer
from utils import mdn_loss


class MLPWithTimeFeatures(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(MLPWithTimeFeatures, self).__init__()
        """
        Two fully connected Layers
        """
        # The first fully connected layer maps the input features to the hidden layer
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        # The second fully connected layer maps the hidden layer features to the outlayer
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout)

        # Initilize weights
        """
        Kaimi_uniform is particularly well-suited for layers using the ReLU activation function. 
        This method helps to avoid issues like vanishing and exploding gradients by appropriately scaling the weights at the start of training.
        """
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu') 
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
    
    def forward(self, mlp_data):       
        x = F.relu(self.fc1(mlp_data))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return x

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size = lstm_hidden_layer_size):
        super(LSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        # The input and output tensor provided to the model will have the format like 'shape(batch, seq, feature)'
        self.lstm = nn.LSTM(input_size, self.hidden_layer_size, num_layers=lstm_layer, batch_first=True, dropout=dropout)

    def forward(self, input_seq):
        # When an lstm processes an input sequence, it outputs a tensor that contains the hidden states for each timestep in sequence.
        # The output tensor has the shape (batch_size, sequence_length, hidden_layer_size)
        batch_size = input_seq.size(0)

        """
        Initializing the hidden state h_0 and cell_state c_0 with zeros ensures that the LSTM starts with a neutral state for each input sequence.
        to(input_seq.device) ensures that all tensors involved in the computation are one the same device
        """
        h_0 = torch.zeros(lstm_layer, batch_size, self.hidden_layer_size).to(input_seq.device)
        c_0 = torch.zeros(lstm_layer, batch_size, self.hidden_layer_size).to(input_seq.device)
        lstm_out, _ = self.lstm(input_seq, (h_0, c_0))
        lstm_out = lstm_out[:, -1, :]  # Hidden state of the last time step

        return lstm_out

class MDNLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_gaussians):
        super(MDNLayer, self).__init__()
        self.num_gaussians = num_gaussians
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)

        # Initializing π with smaller values to encourage uniform initial probabilities
        self.pi = nn.Linear(hidden_dim, num_gaussians)
        nn.init.uniform_(self.pi.weight, -0.1, 0.1)  # Uniform initialization within a small range

        self.sigma = nn.Linear(hidden_dim, num_gaussians * output_dim)
        nn.init.kaiming_normal_(self.sigma.weight)
        nn.init.constant_(self.sigma.bias, 1)  # This sets the initial standard deviations

        self.mu = nn.Linear(hidden_dim, num_gaussians * output_dim)
        nn.init.kaiming_normal_(self.mu.weight)
        
    def forward(self, x):
        # Compute parameters for Gaussian mixtures
        hidden = F.relu(self.hidden_layer(x))
        pi = F.softmax(self.pi(hidden) / 1, dim=1)  # Example temperature T=0.1 to sharpen
        sigma = F.softplus(self.sigma(hidden ))  # Ensures sigma is positive
        mu = F.softplus(self.mu(hidden))  # Ensures mu is positive

        # # Ensuring sigma does not go below a threshold to prevent division by very small numbers
        # sigma = torch.clamp(sigma, min=1e-6)
        return pi, sigma, mu
    
    def regularization_loss(self, l2_lambda=l2_lambda):
        # Calculate L2 regularization for pi, sigma, and mu weights
        reg_loss = 0.0
        for param in [self.pi.weight, self.sigma.weight, self.mu.weight]:
            reg_loss += torch.sum(param ** 2)
        return l2_lambda * reg_loss

class LSTMWithMLP(nn.Module):
    # Here the number of Gaussians determines the flexibility of the model in representing complex distributions.Generally, more Gaussians allow the model to fit more complex and multimodel distribution
    def __init__(self, lstm_input_size=1, output_size=1, num_gaussians=num_gaussians, mlp_input_dim=11, hidden_dim=mdn_hidden_layer): 
        super(LSTMWithMLP, self).__init__()

        self.hidden_layer_size = lstm_hidden_layer_size
        # The input and output tensor provided to the model will have the format like 'shape(batch, seq, feature)'
        self.lstm = LSTM(input_size = lstm_input_size, hidden_layer_size = lstm_hidden_layer_size)
        self.mlp = MLPWithTimeFeatures(input_dim=mlp_input_dim, hidden_dim=mlp_hidden_dim, output_dim=lstm_hidden_layer_size, dropout=dropout)
        self.mdn = MDNLayer(self.hidden_layer_size * 2, hidden_dim, output_size, num_gaussians)

    def forward(self, input_seq, mlp_input):

        lstm_out = self.lstm(input_seq)
        mlp_out = self.mlp(mlp_input)


        lstm_out = (lstm_out - lstm_out.mean(dim=0)) / (lstm_out.std(dim=0) + 1e-6)
        mlp_out = (mlp_out - mlp_out.mean(dim=0)) / (mlp_out.std(dim=0) + 1e-6)
        """
        # The shape of lstm_out: (batch_size, lstm_hidden_dim)
        # The shape of mlp_out: (batch_size, mlp_hidden_dim)
        # After cat, (batch_size, lstm_hidden_dim + mlp_hidden_dim), 
        # This means that for each sample, we merge the features of the LSTM and the features of the MLP together to form a larger feature vector
        """
        combined = torch.cat((lstm_out, mlp_out), dim=1)
        pi, sigma, mu = self.mdn(combined)
        return pi, sigma, mu
    
    def loss(self, pi, sigma, mu, y, l2_lambda):
        # Calculate MDN loss from external file
        mdn_loss_val = mdn_loss(pi, sigma, mu, y) 
        reg_loss = self.mdn.regularization_loss(l2_lambda)
        return mdn_loss_val + reg_loss