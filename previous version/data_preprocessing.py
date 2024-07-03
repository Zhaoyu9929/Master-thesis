import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from feature_engineering import create_sequences
import torch
from config import seq_length

def prepare_mlp_data():
    path = r'C:\Users\yanzh\Desktop\code_and_data\4. Deep learning part\处理数据\MLP data.csv'
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


def standardize_features(data, features):
    scaler = StandardScaler()
    data[features] = scaler.fit_transform(data[features])
    return data, scaler

def one_hot_encode(data, features):
    # One-hot encode categorical features
    encoder = OneHotEncoder(sparse_output=False) # Here return the dense array
    holiday_encoded = encoder.fit_transform(data[[features]])
    holiday_encoded_df = pd.DataFrame(holiday_encoded, columns=encoder.get_feature_names_out([features]))

    # Combined encoded features
    data = pd.concat([data, holiday_encoded_df], axis=1).drop(columns=features)
    return data, encoder

# Normalize the count values
def normalize_count(data):
    scaler = MinMaxScaler()
    data['total_trips'] = scaler.fit_transform(data['total_trips'].values.reshape(-1, 1))
    return data, scaler

def prepare_lstm_data():
    # Load the data
    path = r"C:\Users\yanzh\Desktop\code_and_data\4. Deep learning part\处理数据\2015-2019 total trips.csv"
    data = pd.read_csv(path)

    # Convert date to datetime
    data['date'] = pd.to_datetime(data['date'])

    data, scaler = normalize_count(data)
    data = data['total_trips']
    X, y = create_sequences(data.values, seq_length)

    # Split data into training sand test sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Convert to PyTorch tensor
    # Reshape tensors for LSTM input (batch_size, seq_length, num_features)
    """ 
    Unsquee turn a tensor by adding an extra dimension of depth 1, note that the number of dim
    """
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, scaler

# Unnormalize the valuse
def inverse_normalize_count(data, scaler):
    data = scaler.inverse_transform(data.reshape(-1, 1))
    data = np.round(data)
    return data