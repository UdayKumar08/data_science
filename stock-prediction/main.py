# **Enhancing Stock Market Predictions with Deep Learning Models: An Analysis of LSTM, GRU, CNN, WaveNet, and DeepAR Models**

# ------------------------ Imports ------------------------
import os
import random
import numpy as np
import tensorflow as tf

from data_loading import StockPredictor
from data_processing import DataProcessor
from training import train_model
from testing import evaluate_model
from data_analysis import (
    plot_closing_price,
    plot_moving_averages,
    plot_volatility,
    plot_pairplot,
    plot_model_loss,
    plot_true_vs_pred,
    plot_error_distribution,
    plot_residuals,
    plot_model_comparison
)

from each_models.lstm_model import LSTMHyperModel
from each_models.gru_model import GRUHyperModel
from each_models.cnn_model import CNNHyperModel
from each_models.wavenet_model import WaveNetModel
from each_models.deepar_model import DeepARModel

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ------------------------ Reproducibility ------------------------
def set_seeds():
    os.environ['PYTHONHASHSEED'] = str(1)
    tf.random.set_seed(51)
    np.random.seed(1)
    random.seed(5)

set_seeds()

# ------------------------ Load and Preprocess Data ------------------------
predictor = StockPredictor('data/data.csv')

# Preprocess
processor = DataProcessor(predictor.data)
processed_data, scaler = processor.preprocess_data()
predictor.data = processed_data

# Create sequences
X, y = predictor.create_sequences()

# Split data
X_train, y_train, X_val, y_val, X_test, y_test = predictor.split_data(X, y)

# ------------------------ Basic Data Visualizations ------------------------
plot_closing_price(predictor.data)
plot_moving_averages(predictor.data)
plot_volatility(predictor.data)
plot_pairplot(predictor.data)

# ------------------------ Model Training ------------------------
best_lstm_model, lstm_history = train_model(LSTMHyperModel, X_train, y_train, X_val, y_val, 'LSTM_Model')
best_gru_model, gru_history = train_model(GRUHyperModel, X_train, y_train, X_val, y_val, 'GRU_Model')
best_cnn_model, cnn_history = train_model(CNNHyperModel, X_train, y_train, X_val, y_val, 'CNN_Model')
best_wavenet_model, wavenet_history = train_model(WaveNetModel, X_train, y_train, X_val, y_val, 'WaveNet_Model')
best_deepar_model, deepar_history = train_model(DeepARModel, X_train, y_train, X_val, y_val, 'DeepAR_Model')

# ------------------------ Model Evaluation ------------------------
lstm_metrics = evaluate_model(best_lstm_model, X_test, y_test)
gru_metrics = evaluate_model(best_gru_model, X_test, y_test)
cnn_metrics = evaluate_model(best_cnn_model, X_test, y_test)
wavenet_metrics = evaluate_model(best_wavenet_model, X_test, y_test)
deepar_metrics = evaluate_model(best_deepar_model, X_test, y_test)

print("LSTM Metrics:", lstm_metrics)
print("GRU Metrics:", gru_metrics)
print("CNN Metrics:", cnn_metrics)
print("WaveNet Metrics:", wavenet_metrics)
print("DeepAR Metrics:", deepar_metrics)

# ------------------------ Compare All Models ------------------------
models = ['LSTM', 'GRU', 'CNN', 'WaveNet', 'DeepAR']
metrics = ['Test Loss', 'MAE', 'MSE', 'RMSE', 'R²']

performance = pd.DataFrame(
    [lstm_metrics, gru_metrics, cnn_metrics, wavenet_metrics, deepar_metrics],
    index=models,
    columns=metrics
)

print(performance)

# ------------------------ Plot Loss Curves ------------------------
plot_model_loss(lstm_history, 'LSTM')
plot_model_loss(gru_history, 'GRU')
plot_model_loss(cnn_history, 'CNN')
plot_model_loss(wavenet_history, 'WaveNet')
plot_model_loss(deepar_history, 'DeepAR')

# ------------------------ True vs Predicted Plots ------------------------
y_true = y_test.flatten()

lstm_predictions = best_lstm_model.predict(X_test).flatten()
gru_predictions = best_gru_model.predict(X_test).flatten()
cnn_predictions = best_cnn_model.predict(X_test).flatten()
wavenet_predictions = best_wavenet_model.predict(X_test).flatten()
deepar_predictions = best_deepar_model.predict(X_test).flatten()

plot_true_vs_pred(y_true, lstm_predictions, 'LSTM', color_true='yellow', color_pred='royalblue')
plot_true_vs_pred(y_true, gru_predictions, 'GRU', color_true='orange', color_pred='green')
plot_true_vs_pred(y_true, cnn_predictions, 'CNN', color_true='orange', color_pred='red')
plot_true_vs_pred(y_true, wavenet_predictions, 'WaveNet', color_true='yellow', color_pred='purple')
plot_true_vs_pred(y_true, deepar_predictions, 'DeepAR', color_true='yellow', color_pred='grey')

# ------------------------ Error and Residual Analysis ------------------------
errors = {
    'LSTM': y_true - lstm_predictions,
    'GRU': y_true - gru_predictions,
    'CNN': y_true - cnn_predictions,
    'WaveNet': y_true - wavenet_predictions,
    'DeepAR': y_true - deepar_predictions
}

plot_error_distribution(errors)
plot_residuals(errors)

# ------------------------ Full Comparison Plot ------------------------
predictions_dict = {
    'LSTM': lstm_predictions,
    'GRU': gru_predictions,
    'CNN': cnn_predictions,
    'WaveNet': wavenet_predictions,
    'DeepAR': deepar_predictions
}

time_steps = list(range(len(y_true)))
plot_model_comparison(time_steps, y_true, predictions_dict)
