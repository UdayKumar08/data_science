from keras_tuner import RandomSearch  # For hyperparameter tuning of models
from tensorflow.keras.callbacks import EarlyStopping  # For stopping training when the model stops improving

""" 
Define a function to train a given hypermodel using the training and validation data 
"""
def train_model(hypermodel, X_train, y_train, X_val, y_val, model_name):
    """ Initialize a RandomSearch tuner for hyperparameter tuning of the given model """
    tuner = RandomSearch(
        hypermodel(input_shape=X_train.shape[1:]),  # Initialize the hypermodel with the input shape
        objective='val_loss',  # Objective function to minimize during tuning
        max_trials=3,  # Maximum number of different hyperparameter combinations to try
        executions_per_trial=1,  # Number of times to train the model with each set of hyperparameters
        directory='stock_predictions',  # Directory to save the results
        project_name=model_name  # Name of the project
    )

    """ Perform hyperparameter search by training the model """
    tuner.search(
        X_train, y_train,  # Training data
        epochs=20,  # Number of epochs for each trial
        validation_data=(X_val, y_val),  # Validation data
        callbacks=[EarlyStopping(monitor='val_loss', patience=5)]  # Stop training if validation loss does not improve for 5 epochs
    )

    """ Get the best model based on the search """
    best_model = tuner.get_best_models(num_models=1)[0]  # Retrieve the best model from the tuner

    """ Fine-tune the best model """
    history = best_model.fit(
        X_train, y_train,  # Training data
        epochs=10,  # Number of epochs for fine-tuning
        batch_size=32,  # Batch size for training
        validation_data=(X_val, y_val),  # Validation data
        verbose=1  # Print training progress
    )
    
    return best_model, history  # Return the best model and its training history
