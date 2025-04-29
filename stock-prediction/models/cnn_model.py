from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam
from keras_tuner import HyperModel

""" 
Define a HyperModel class for tuning a Convolutional Neural Network (CNN) model 
"""
class CNNHyperModel(HyperModel):
    """ Initialize the CNNHyperModel with input shape """
    def __init__(self, input_shape):
        self.input_shape = input_shape

    """ Build the CNN model with hyperparameters to tune """
    def build(self, hp):
        """ Define the CNN model architecture with tunable hyperparameters """
        model = Sequential([
            Input(shape=self.input_shape),
            Conv1D(
                filters=hp.Int('filters', 32, 128, step=32),
                kernel_size=hp.Int('kernel_size', 3, 7, step=2),
                activation='relu'
            ),
            MaxPooling1D(pool_size=2),
            Conv1D(
                filters=hp.Int('filters', 32, 128, step=32),
                kernel_size=hp.Int('kernel_size', 3, 7, step=2),
                activation='relu'
            ),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(50, activation='relu'),
            Dense(1)
        ])
        """ Compile the model with Adam optimizer, mean squared error loss, and evaluation metrics """
        model.compile(
            optimizer=Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG')),
            loss='mean_squared_error',
            metrics=['mean_absolute_error', 'mean_squared_error']
        )
        return model
