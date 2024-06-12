import numpy as np

def extract_time_features(data):
    data['hour'] = data['date'].dt.hour
    data['month'] = data['date'].dt.month
    return data

def prepare_mlp_input(data, features):
    return data[features]

# Create sequences
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)
