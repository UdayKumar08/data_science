import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
from sklearn.preprocessing import MinMaxScaler  # For scaling data to a range between 0 and 1

class StockPredictor:
    """Class to handle loading, processing, and preparing stock data."""
    
    def __init__(self, filename):
        """Initialize the predictor by loading data from a CSV file."""
        self.load_data(filename)
    
    def load_data(self, filename):
        """Load and preprocess data from a file."""
        try:
            self.data = pd.read_csv(filename)
            self.data['Date'] = pd.to_datetime(self.data['Date'], errors='coerce')
            self.data.dropna(subset=['Date'], inplace=True)
            self.data.set_index('Date', inplace=True)
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def create_sequences(self, sequence_length=10):
        """Create sequences from time-series data for LSTM training."""
        columns_to_use = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'EMA_10', 'Volatility', 'Lag_1', 'Lag_2']
        xs, ys = [], []
        for i in range(len(self.data) - sequence_length):
            x = self.data.iloc[i:(i + sequence_length)][columns_to_use].values
            y = self.data.iloc[i + sequence_length]['Close']
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def split_data(self, X, y, train_ratio=0.7, val_ratio=0.15):
        """Split data into training, validation, and testing sets."""
        train_end = int(len(X) * train_ratio)
        val_end = int(len(X) * (train_ratio + val_ratio))
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        return X_train, y_train, X_val, y_val, X_test, y_test
