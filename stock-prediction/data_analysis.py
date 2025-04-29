import matplotlib.pyplot as plt  # For plotting graphs
import seaborn as sns  # For statistical plots
import pandas as pd  # For handling data
import numpy as np  # For calculations

""" 
Define a function to plot closing price over time 
"""
def plot_closing_price(data):
    plt.figure(figsize=(14, 7))
    plt.plot(data['Close'], label='Closing Price')
    plt.title('Closing Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

""" 
Define a function to plot Closing Price with Moving Averages 
"""
def plot_moving_averages(data):
    plt.figure(figsize=(14, 7))
    sns.set(style="whitegrid")
    sns.lineplot(data=data, x=data.index, y='Close', label='Closing Price', linewidth=2.5)
    sns.lineplot(data=data, x=data.index, y='SMA_10', label='10-Day SMA', linewidth=2.5)
    sns.lineplot(data=data, x=data.index, y='EMA_10', label='10-Day EMA', linewidth=2.5)
    plt.title('Closing Price with SMA and EMA')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(title='Legend')
    plt.show()

""" 
Define a function to plot volatility over time 
"""
def plot_volatility(data):
    plt.figure(figsize=(14, 7))
    plt.plot(data['Volatility'], label='Volatility')
    plt.title('Volatility Over Time')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.legend()
    plt.show()

""" 
Define a function to plot pairplot 
"""
def plot_pairplot(data):
    sns.pairplot(data[['Close', 'Volume', 'SMA_10', 'EMA_10', 'Volatility']])
    plt.show()

""" 
Define a function to plot model loss curves 
"""
def plot_model_loss(history, model_name):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.savefig(f'images/{model_name.lower()}_model_loss.png', dpi=100, bbox_inches='tight')
    plt.show()

""" 
Define a function to plot True vs Predicted values for a model 
"""
def plot_true_vs_pred(y_true, y_pred, model_name, color_true='yellow', color_pred='blue'):
    time_steps = list(range(len(y_true)))
    plt.figure(figsize=(16, 8))
    plt.plot(time_steps, y_true, label='True Values', color=color_true, linestyle='-')
    plt.plot(time_steps, y_pred, label=f'{model_name} Predictions', color=color_pred, linestyle='--')
    plt.title(f'{model_name}: True Values vs Predictions')
    plt.xlabel('Time')
    plt.ylabel('Normalized Stock Price')
    plt.legend(loc='upper right')
    plt.savefig(f'images/{model_name.lower()}_true_vs_pred.png', dpi=100, bbox_inches='tight')
    plt.show()

""" 
Define a function to plot error distribution for each model 
"""
def plot_error_distribution(errors):
    plt.figure(figsize=(12, 8))
    for model, error in errors.items():
        sns.histplot(error, bins=30, kde=True, label=model)
    plt.title('Error Distribution for Each Model')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('images/error_distribution.png', dpi=100, bbox_inches='tight')
    plt.show()

""" 
Define a function to plot residual plots for each model 
"""
def plot_residuals(errors):
    plt.figure(figsize=(12, 8))
    for model, error in errors.items():
        plt.scatter(range(len(error)), error, label=model, alpha=0.5)
    plt.title('Residual Plot for Each Model')
    plt.xlabel('Data Point Index')
    plt.ylabel('Residuals (Error)')
    plt.legend()
    plt.savefig('images/residual_plots.png', dpi=100, bbox_inches='tight')
    plt.show()

""" 
Define a function to plot comparison of True Values vs Predictions for all models 
"""
def plot_model_comparison(time_steps, y_true, predictions_dict):
    plt.figure(figsize=(14, 10))
    plt.plot(time_steps, y_true, label='True Values', color='blue', linestyle='-', linewidth=2)
    for model_name, y_pred in predictions_dict.items():
        plt.plot(time_steps, y_pred, label=f'{model_name} Predictions', linestyle='--', linewidth=1.5)
    plt.title('Comparison of True Values vs Predictions for All Models', fontsize=16)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Normalized Stock Price', fontsize=14)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('images/model_comparison_true_vs_pred.png', dpi=300, bbox_inches='tight')
    plt.savefig('images/model_comparison_true_vs_pred.pdf', dpi=300, bbox_inches='tight')
    plt.show()
