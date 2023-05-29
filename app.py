from flask import Flask, jsonify, render_template
from ts_models import multivariate_ts_lstm
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

if __name__ == '__main__':
    app.run(debug=True)