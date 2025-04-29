from sklearn.preprocessing import MinMaxScaler  # For scaling data to a range between 0 and 1

class DataProcessor:
    """Class to handle preprocessing tasks for stock data."""

    def __init__(self, data):
        self.data = data
        self.scaler = None

    def remove_currency_symbols(self):
        """Convert price-related columns to floats by removing currency symbols."""
        price_columns = ['Close/Last', 'Open', 'High', 'Low']
        for col in price_columns:
            if col in self.data.columns:
                self.data[col] = self.data[col].str.replace('$', '').astype(float)
        self.data.rename(columns={'Close/Last': 'Close'}, inplace=True)

    def create_features(self):
        """Generate technical indicators as additional features."""
        self.data['SMA_10'] = self.data['Close'].rolling(window=10).mean()
        self.data['EMA_10'] = self.data['Close'].ewm(span=10, adjust=False).mean()
        self.data['Volatility'] = self.data['Close'].rolling(window=10).std()
        self.data['Lag_1'] = self.data['Close'].shift(1)
        self.data['Lag_2'] = self.data['Close'].shift(2)
        self.data.dropna(inplace=True)

    def normalize_features(self):
        """Scale features to a range suitable for neural network inputs."""
        self.scaler = MinMaxScaler()
        features_to_scale = ['Close', 'Volume', 'Open', 'High', 'Low', 'SMA_10', 'EMA_10', 'Volatility', 'Lag_1', 'Lag_2']
        self.data[features_to_scale] = self.scaler.fit_transform(self.data[features_to_scale])

    def preprocess_data(self):
        """Run all preprocessing steps."""
        self.remove_currency_symbols()
        self.create_features()
        self.normalize_features()
        return self.data, self.scaler