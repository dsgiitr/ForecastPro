from flask import Flask, jsonify, render_template, request
from ts_models import multivariate_ts_lstm, univariate_ts_rnn
import pandas as pd
import numpy as np
import sklearn
import json
import pickle

app = Flask(__name__)

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
    print(df)
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

    return past_test_val.flatten(), past_preds.flatten(), final_forecast.flatten()

# Define the route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/data', methods=['GET', 'POST'])
def data():
    # Read the contents of test.json
    with open(request.json['testfile'], 'r') as test_file:
        test_data = json.load(test_file)

    # Read the contents of predictions.json
    with open(request.json['predfile'], 'r') as predictions_file:
        predictions_data = json.load(predictions_file)

    test_data = test_data[-len(predictions_data):]

    # Access the data as needed
    # For example, convert the data to lists and return as JSON
    return jsonify(test=test_data, predictions=predictions_data)


@app.route('/train-model', methods=['GET', 'POST'])
def train_model():
    past_test_val, past_preds, final_forecast = fetch_data(request.form['stocks'], 10, 10, 20)
    
    # convert all y cooridnates into a a list of dicts with {x, y} for plotting in chart.js
    past_test_val_dict  = [{'x':str(x), 'y':str(y)} for x, y in zip(range(len(past_test_val)), past_test_val)]
    past_preds_dict     = [{'x':str(x), 'y':str(y)} for x, y in zip(range(len(past_preds)), past_preds)]
    final_forecast_dict = [{'x':str(x), 'y':str(y)} for x, y in zip(range(len(past_preds), len(past_preds) + len(final_forecast)), final_forecast)]

    print(past_test_val_dict)
    print(past_preds_dict)
    print(final_forecast_dict)


    return jsonify(test=past_test_val_dict, predictions=past_preds_dict, forecast=final_forecast_dict)

if __name__ == '__main__':
    app.run(debug=False, port=5001)
