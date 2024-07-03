import configparser

config = configparser.ConfigParser()
config.read('config.ini')

# Data parameters
seq_length = config.getint('MODEL', 'seq_length')

# Model parameters
batch_size = config.getint('MODEL', 'batch_size')
epochs = config.getint('MODEL', 'epochs')
learning_rate = config.getfloat('MODEL', 'learning_rate')
num_gaussians = config.getint('MODEL', 'num_gaussians')

# LSTM parameters
lstm_hidden_layer_size = config.getint('MODEL', 'lstm_hidden_layer_size')
dropout = config.getfloat('MODEL', 'dropout') 

# MLP parameters
mlp_hidden_dim = config.getint('MODEL', 'mlp_hidden_dim')

