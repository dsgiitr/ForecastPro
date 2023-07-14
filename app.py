from flask import Flask, jsonify, render_template
from ts_models import univariate_ts_rnn
import pandas as pd
import numpy as np
import sklearn
import json
import pickle

app = Flask(__name__)

# Define the route for the home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def data():
    # Read the contents of test.json
    with open('test.json', 'r') as test_file:
        test_data = json.load(test_file)

    # Read the contents of predictions.json
    with open('predictions.json', 'r') as predictions_file:
        predictions_data = json.load(predictions_file)

    test_data = test_data[-len(predictions_data):]    

    # Access the data as needed
    # For example, convert the data to lists and return as JSON
    return jsonify(test=test_data, predictions=predictions_data)


# Getting predictions from the model
def fetch_data(stock_name='AMZN', forecast_steps=10, time_steps=10, epochs=400):
    model = univariate_ts_rnn(
        time_steps=time_steps,
        epochs=epochs,
        loss_stopping_patience=20,
        batch_size=15,
        loss_curve=True,
        forecast_eval=True,
        split=0.75,
        learning_rate=0.001,
        return_predictions=True
    )

    # fetching data
    df = pd.read_csv('data/portfolio_data.csv')
    series = np.array(df[stock_name])

    scaler, model, past_test_val, past_preds = model.train(data=series)
    past_test_val = np.asarray(past_test_val)
    past_preds = np.asarray(past_preds)

    last_batch = series[len(series) - time_steps:]
    forecast = []
    for step in range(forecast_steps):
        pred = model.predict(np.array(last_batch).reshape(1, time_steps, 1))
        forecast.append(pred)
        last_batch = np.append(last_batch[1:], pred)

    forecast = np.array(forecast).flatten()
    final_forecast = scaler.inverse_transform(forecast.reshape(-1, 1))

    return past_test_val, past_preds, final_forecast


if __name__ == '__main__':
    app.run(debug=True)