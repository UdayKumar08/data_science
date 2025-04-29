from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Activation, Input
from tensorflow.keras.optimizers import Adam
from keras_tuner import HyperModel

""" 
Define a HyperModel class for tuning a WaveNet model 
"""
class WaveNetModel(HyperModel):
    """ Initialize the WaveNetModel with input shape """
    def __init__(self, input_shape):
        self.input_shape = input_shape

    """ Build the WaveNet model with hyperparameters to tune """
    def build(self, hp):
        """Define the WaveNet model architecture with tunable hyperparameters """
        model = Sequential([
            Input(shape=self.input_shape),
            
            Conv1D(
                filters=hp.Int('filters_1', 32, 128, step=32),
                kernel_size=2,
                dilation_rate=1,
                padding='causal'
            ),
            Activation('relu'),

            Conv1D(
                filters=hp.Int('filters_2', 32, 128, step=32),
                kernel_size=2,
                dilation_rate=2,
                padding='causal'
            ),
            Activation('relu'),

            Conv1D(
                filters=hp.Int('filters_3', 32, 128, step=32),
                kernel_size=2,
                dilation_rate=4,
                padding='causal'
            ),
            Activation('relu'),

            Flatten(),
            Dense(units=hp.Int('units', 50, 200, step=50)),
            Dense(1)
        ])
        
        """ Compile the model with Adam optimizer, mean squared error loss, and evaluation metrics"""
        model.compile(
            optimizer=Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG')),
            loss='mean_squared_error',
            metrics=['mean_absolute_error']
        )
        return model
