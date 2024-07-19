import configparser

# Helper function to convert configuration string to list of integers
def parse_list_of_ints(config, section, option):
    value_str = config.get(section, option)
    # Remove the brackets and split the string into a list
    int_list = [int(x.strip()) for x in value_str.strip('[]').split(',')]
    return int_list

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
lstm_hidden_layer_size = parse_list_of_ints(config, 'MODEL', 'lstm_hidden_layer_size')
dropout = config.getfloat('MODEL', 'dropout') 

# MLP parameters
mlp_hidden_dim = config.getint('MODEL', 'mlp_hidden_dim')

# EarlyStopping
patience = config.getint('MODEL', 'patience')
delta = config.getfloat('MODEL', 'delta') 