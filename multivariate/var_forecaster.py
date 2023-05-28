import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

def forecast_with_confidence_intervals(data, steps_ahead, confidence_level=0.95):
    model = sm.tsa.VAR(data)
    results = model.fit()
    predictions = results.forecast(steps_ahead)
    prediction_mean = predictions.mean(axis=1)
    prediction_std = predictions.std(axis=1)
    lower_bound = prediction_mean - (1.96 * prediction_std)
    upper_bound = prediction_mean + (1.96 * prediction_std)
    lower_bound = np.maximum(lower_bound, 0)
    x = np.arange(len(data), len(data) + steps_ahead)
    plt.fill_between(x, lower_bound, upper_bound, alpha=0.2)
    plt.plot(x, prediction_mean, 'r', label='prediction')
    plt.legend()
    plt.show()
