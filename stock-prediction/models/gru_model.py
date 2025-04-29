from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from keras_tuner import HyperModel

""" 
Define a HyperModel class for tuning a Gated Recurrent Unit (GRU) model 
"""
class GRUHyperModel(HyperModel):
    """ Initialize the GRUHyperModel with input shape """
    def __init__(self, input_shape):
        self.input_shape = input_shape

    """ Build the GRU model with hyperparameters to tune """
    def build(self, hp):
        """ Define the GRU model architecture with tunable hyperparameters """
        model = Sequential([
            Input(shape=self.input_shape),
            GRU(units=hp.Int('units', 50, 200, step=50), return_sequences=True),
            Dropout(hp.Float('dropout_rate', 0.1, 0.5, step=0.1)),
            GRU(units=hp.Int('units', 50, 200, step=50)),
            Dropout(hp.Float('dropout_rate', 0.1, 0.5, step=0.1)),
            Dense(1)
        ])
        """ Compile the model with Adam optimizer, mean squared error loss, and evaluation metrics """
        model.compile(
            optimizer=Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG')),
            loss='mean_squared_error',
            metrics=['mean_absolute_error', 'mean_squared_error']
        )
        return model
