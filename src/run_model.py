
# FETCHING DATA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import os
import pickle
import json
import yfinance as yf
import pandas as pd
import tqdm
from tqdm import tqdm

tickers = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'JPM', 'V', 'JNJ', 'PG', 'XOM', 'T', 'BAC', 'WMT', 'INTC', 'PFE',
           'VZ', 'KO', 'TSLA', 'MRK', 'DIS', 'UNH', 'HD', 'ADBE', 'CMCSA', 'PEP', 'CSCO', 'NVDA', 'NFLX',
           'ABT', 'NKE', 'CVX', 'ACN', 'TMUS', 'BMY', 'LLY', 'TMO', 'IBM', 'MCD', 'ORCL', 'UPS', 'MDT', 'COST',
           'PM', 'AVGO', 'SAP', 'HON', 'NEE', 'TXN', 'MO']

data = yf.download(tickers, period="1000d",group_by='ticker', auto_adjust=True)

open_prices = pd.DataFrame({ticker: data[ticker]['Open'] for ticker in tickers})
open_prices.to_csv('LSTNet\data\large_portfolio.csv')
np.savetxt('LSTNet\data\large_portfolio.txt', np.array(open_prices), delimiter=',')


import subprocess

def train_model(
                data_path="data/large_portfolio.txt",
                horizon=1,
                save_name="large_portfolio",
                window=28,
                validpercent=0.40,
                batchsize=16,
                skip=7,
                epochs=20,
                n_series=6,
                learning_rate=0.001,
                dropout=0.2,
                highway=7,
                GRUUnits=100,
                SkipGRUUnits=5,

                ):
    os.chdir("LSTNet")
    command = f'python main.py --data={data_path} --horizon={horizon} --save="save/{save_name}_horizon{horizon}_window{window}_skip{skip}" --test --savehistory --logfilename="log/lstnet" --debuglevel=20 --predict="all" --plot --save-plot="save/{save_name}_horizon{horizon}_window{window}_skip{skip}_plots" --window={window} --validpercent={validpercent} --batchsize={batchsize} --skip={skip} --epochs={epochs} --CNNKernel={n_series} --lr={learning_rate} --dropout={dropout} --highway={highway} --GRUUnits={GRUUnits} --SkipGRUUnits={SkipGRUUnits}' 
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()

    if process.returncode == 0:
        print(output.decode())
    else:
        print("Error:", error.decode())

train_model()