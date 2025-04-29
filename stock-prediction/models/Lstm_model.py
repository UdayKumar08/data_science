from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from keras_tuner import HyperModel

class LSTMHyperModel(HyperModel):
    """Hypermodel for tuning the LSTM neural network."""
    
    def __init__(self, input_shape):
        """Initialize with the shape of data input to the LSTM."""
        self.input_shape = input_shape
    
    def build(self, hp):
        """Build LSTM model with tunable hyperparameters."""
        model = Sequential([
            Input(shape=self.input_shape),
            LSTM(units=hp.Int('units', 50, 200, step=50), return_sequences=True),
            Dropout(hp.Float('dropout_rate', 0.1, 0.5, step=0.1)),
            LSTM(units=hp.Int('units', 50, 200, step=50)),
            Dropout(hp.Float('dropout_rate', 0.1, 0.5, step=0.1)),
            Dense(1)
        ])
        model.compile(optimizer=Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG')), loss='mean_squared_error')
        return model
