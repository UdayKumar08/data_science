from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from keras_tuner import HyperModel

""" 
Define a HyperModel class for tuning a DeepAR model 
"""
class DeepARModel(HyperModel):
    """ Initialize the DeepARModel with input shape """
    def __init__(self, input_shape):
        self.input_shape = input_shape

    """Build the DeepAR model with hyperparameters to tune """
    def build(self, hp):
        """ Define the input layer for the DeepAR model """
        inputs = Input(shape=self.input_shape)
        
        """ Define the first LSTM layer with tunable number of units and return sequences """
        x = LSTM(units=hp.Int('units', 50, 200, step=50), return_sequences=True)(inputs)
        
        """ Add a Dropout layer with a tunable dropout rate for regularization """
        x = Dropout(hp.Float('dropout_rate', 0.1, 0.5, step=0.1))(x)
        
        """ Define the second LSTM layer with tunable number of units """
        x = LSTM(units=hp.Int('units', 50, 200, step=50))(x)
        
        """ Add another Dropout layer with a tunable dropout rate for regularization """
        x = Dropout(hp.Float('dropout_rate', 0.1, 0.5, step=0.1))(x)
        
        """ Define the output layer with a single neuron for regression """
        outputs = Dense(1)(x)
        
        """  Create the model using the functional API """
        model = Model(inputs, outputs)
        
        """ Compile the model with Adam optimizer, mean squared error loss, and mean absolute error metric """
        model.compile(
            optimizer=Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG')),
            loss='mean_squared_error',
            metrics=['mean_absolute_error']
        )
        
        return model
