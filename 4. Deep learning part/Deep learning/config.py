import configparser

config = configparser.ConfigParser()
config.read('config.ini')

# Model parameters
batch_size = config.getint('MODEL', 'batch_size')
epochs = config.getint('MODEL', 'epochs')
learning_rate = config.getfloat('MODEL', 'learning_rate')
num_gaussians = config.getint('MODEL', 'num_gaussians')

# Data parameters
seq_length = config.getint('MODEL', 'seq_length')