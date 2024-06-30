import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
import torch
from config import seq_length
import math

def prepare_mlp_data():
    path = r"C:\Users\yanzh\Desktop\code_and_data\4. Deep learning part\MLP data.csv"
    data = pd.read_csv(path)
    
    # Extract time features and prepare MLP input, hour and month
    data['date'] = pd.to_datetime(data['date'])
    data['hour'] = data['date'].dt.hour
    data['month'] = data['date'].dt.month

    # Standardize feartures, it is applicable if the data follows the normal distribution
    numeric_features1 = ['temperature_2m', 'CRASH COUNT']
    scaler1 = StandardScaler()
    data[numeric_features1] = scaler1.fit_transform(data[numeric_features1])
    
    # MinMaxScaler
    numeric_features2 = ['precipitation', 'rain', 'snowfall', 'snow_depth', 'wind_speed_10m']
    scaler2 = MinMaxScaler()
    data[numeric_features2] = scaler2.fit_transform(data[numeric_features2])
    
    # One_hot encode holiday feature
    data, _ = one_hot_encode(data, 'holiday')
    
    # Prepare MLP input
    feature_columns = ['hour', 'month'] + numeric_features1 + numeric_features2 + list(data.filter(regex='holiday').columns)
    data = data[feature_columns]
    
    # Ensure all data is numeric
    data = data.apply(pd.to_numeric, errors='coerce')
    return data
    
# For holidat information
def one_hot_encode(data, features):
    # One-hot encode categorical features
    encoder = OneHotEncoder(sparse_output=False) # Here return the dense array
    holiday_encoded = encoder.fit_transform(data[[features]])
    holiday_encoded_df = pd.DataFrame(holiday_encoded, columns=encoder.get_feature_names_out([features]))

    # Combined encoded features
    data = pd.concat([data, holiday_encoded_df], axis=1).drop(columns=features)
    return data, encoder

def prepare_lstm_data():
    # Load the data
    path = r"C:\Users\yanzh\Desktop\code_and_data\4. Deep learning part\2015-2019 total trips.csv"
    
    data = pd.read_csv(path)

    # Convert date to datetime
    data['date'] = pd.to_datetime(data['date'])
    data['hour'] = data['date'].dt.hour

    # Sort data by date and hour
    data = data.sort_values(by=['date', 'hour']).reset_index(drop=True)
    data, scaler = normalize_count(data)
    data = data['total_trips']
    windows = sliding_window_cross_validation(data, seq_length)
    return windows, scaler

# Normalize the count values
def normalize_count(data):
    scaler = MinMaxScaler()
    data['total_trips'] = scaler.fit_transform(data['total_trips'].values.reshape(-1, 1))
    return data, scaler

# Create sequences
def create_sequences(data, seq_length):
    xs, ys = [], []
    data = data.reset_index(drop=True)
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

# Sliding window cross validation 
def sliding_window_cross_validation(data, seq_length):
    num_windows = 5
    window_size = math.ceil(len(data) // (num_windows - 1.2))
    step_size = math.ceil(window_size * 0.7)

    windows = []
    start = 0
    while start + step_size <= len(data):
        end = start + window_size
        window_data = data[start:end]
        X, y = create_sequences(window_data, seq_length)
        
        train_val_split = int(len(X) * 0.9)
        X_train, X_test = X[:train_val_split], X[train_val_split:]
        y_train, y_test = y[:train_val_split], y[train_val_split:]

        # Convert to PyTorch tensor
        # Reshape tensors for LSTM input (batch_size, seq_length, num_features)
        """ 
        Unsquee turn a tensor by adding an extra dimension of depth 1, note that the number of dim
        """
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        
        windows.append((X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor))
        start += step_size

        print(f"Total windows created: {len(windows)}")
    return windows

# Unnormalize the valuse
def inverse_normalize_count(data, scaler):
    data = scaler.inverse_transform(data.reshape(-1, 1))
    data = np.round(data)
    return data