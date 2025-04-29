import numpy as np  # For numerical operations
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # For evaluation metrics

""" 
Define a function to evaluate the trained model using test data 
"""
def evaluate_model(model, X_test, y_test):
    """ Evaluate the model on the test data and obtain the loss and metrics """
    results = model.evaluate(X_test, y_test, verbose=1)  # Evaluate the model and get results

    """ Unpack the results if it's a list """
    if isinstance(results, list):  # Check if the results are returned as a list
        test_loss = results[0]  # Test loss is the first element
        mae = results[1] if len(results) > 1 else mean_absolute_error(y_test, model.predict(X_test))  # Mean Absolute Error
        mse = results[2] if len(results) > 2 else mean_squared_error(y_test, model.predict(X_test))  # Mean Squared Error
    else:
        test_loss = results  # If not a list, assume it's the test loss
        mae = mean_absolute_error(y_test, model.predict(X_test))  # Calculate Mean Absolute Error
        mse = mean_squared_error(y_test, model.predict(X_test))  # Calculate Mean Squared Error

    """ Calculate additional metrics for model evaluation """
    predictions = model.predict(X_test)  # Get the predictions of the model on the test data
    rmse = np.sqrt(mse)  # Calculate Root Mean Squared Error
    r2 = r2_score(y_test, predictions)  # Calculate R-squared (coefficient of determination)

    return test_loss, mae, mse, rmse, r2  # Return all the calculated metrics
