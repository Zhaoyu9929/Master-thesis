import torch
import torch.nn as nn
import torch.nn.functional as F
from data_preprocessing import prepare_mlp_data
from config import num_gaussians, lstm_hidden_layer_size, dropout, mlp_hidden_dim
import pandas as pd


class MLPWithTimeFeatures(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPWithTimeFeatures, self).__init__()

        # The first fully connected layer maps the input features to the hidden layer
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        # The second fully connected layer maps the hidden layer features to the outlayer
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, mlp_data):       

        x = F.relu(self.fc1(mlp_data))
        x = F.relu(self.fc2(x))
        return x


class MDNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_gaussians):
        super(MDNLayer, self).__init__()
        self.num_gaussians = num_gaussians

        # Establish three fully connected layers from input_dim to num_gaussians
        # Caculate the weights of the mixture of Gaussian distribution
        self.z_pi = nn.Linear(input_dim, num_gaussians)
        self.z_sigma = nn.Linear(input_dim, num_gaussians * output_dim) # Here actually output_dim is 1 beacuse it is the dimension of each Gaussian distribution ouput
        self.z_mu = nn.Linear(input_dim, num_gaussians * output_dim)

    # Here the keyword self in python is used to represent the instance of the class. It allow you to access attributes and methods of the class in object-oriented programming.
    # General Explanation of self: self is used to refer to the current instance of the class / By useing self, you can accsee other access other methods and attributes with the same class.
    def forward(self, x):
        # The softmax function is used to convert a vector of valuse into a probability distribution, where the sum of all probabilities is 
        pi = F.softmax(self.z_pi(x), dim=1) # dim=1 is because we ensure that the softmax funtion is applied independently to each sample in the batch.
        sigma = torch.exp(self.z_sigma(x)) # this ensures that the output standard sigman is always postive.
        mu = self.z_mu(x)        
        return pi, sigma, mu

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size = lstm_hidden_layer_size):
        super(LSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        # The input and output tensor provided to the model will have the format like 'shape(batch, seq, feature)'
        self.lstm = nn.LSTM(input_size, self.hidden_layer_size, num_layers=2, batch_first=True, dropout=dropout)

    def forward(self, input_seq):

        # When an lstm processes an input sequence, it outputs a tensor that contains the hidden states for each timestep in sequence.
        # The output tensor has the shape (batch_size, sequence_length, hidden_layer_size)
        batch_size = input_seq.size(0)
        h_0 = torch.zeros(2, batch_size, self.hidden_layer_size).to(input_seq.device)
        c_0 = torch.zeros(2, batch_size, self.hidden_layer_size).to(input_seq.device)
        lstm_out, _ = self.lstm(input_seq, (h_0, c_0))

        lstm_out = lstm_out[:, -1, :]  # Hidden state of the last time step

        return lstm_out


class LSTMWithMLP(nn.Module):
    # Here the number of Gaussians determines the flexibility of the model in representing complex distributions.Generally, more Gaussians allow the model to fit more complex and multimodel distribution
    def __init__(self, input_size=1, output_size=1, num_gaussians=num_gaussians, mlp_input_dim=9): 
        super(LSTMWithMLP, self).__init__()

        self.hidden_layer_size = lstm_hidden_layer_size
        # The input and output tensor provided to the model will have the format like 'shape(batch, seq, feature)'
        self.lstm = LSTM(input_size=1, hidden_layer_size = lstm_hidden_layer_size)
        self.mlp = MLPWithTimeFeatures(input_dim=mlp_input_dim, hidden_dim=mlp_hidden_dim, output_dim=lstm_hidden_layer_size)
        self.mdn = MDNLayer(self.hidden_layer_size * 2, output_size, num_gaussians)

    def forward(self, input_seq, mlp_input):

        lstm_out = self.lstm(input_seq)
        mlp_out = self.mlp(mlp_input)

        # The shape of lstm_out: (batch_size, lstm_hidden_dim)
        # The shape of mlp_out: (batch_size, mlp_hidden_dim)
        # After cat, (batch_size, lstm_hidden_dim + mlp_hidden_dim), This means that for each sample, we merge the features of the LSTM and the features of the MLP together to form a larger feature vector
        combined = torch.cat((lstm_out, mlp_out), dim=1)
        pi, sigma, mu = self.mdn(combined)
        return pi, sigma, mu