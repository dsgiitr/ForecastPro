import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import os
import pickle
import json
import tqdm
from tqdm import tqdm
from tensorflow import keras
from keras.models import model_from_json
from LSTNet.lstnet_util import GetArguments, LSTNetInit
from LSTNet.lstnet_model import PreSkipTrans, PostSkipTrans, PreARTrans, PostARTrans, LSTNetModel, ModelCompile
import subprocess
import yfinance as yf
import pandas as pd

class FetchStocks():
    def __init__(self,
                 period="1000d",
                 tickers=['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'JPM', 'V', 'JNJ', 'PG', 'XOM', 'T', 'BAC', 'WMT', 'INTC', 'PFE',
           'VZ', 'KO', 'TSLA', 'MRK', 'DIS', 'UNH', 'HD', 'ADBE', 'CMCSA', 'PEP', 'CSCO', 'NVDA', 'NFLX',
           'ABT', 'NKE', 'CVX', 'ACN', 'TMUS', 'BMY', 'LLY', 'TMO', 'IBM', 'MCD', 'ORCL', 'UPS', 'MDT', 'COST',
           'PM', 'AVGO', 'SAP', 'HON', 'NEE', 'TXN', 'MO'],
           save_files=True
           ):
        self.csv_path="LSTNet\data\large_portfolio.csv"
        self.txt_path="LSTNet\data\large_portfolio.txt"
        self.tickers=tickers
        self.period=period
        self.downloaded_data = yf.download(self.tickers, period=self.period,group_by='ticker', auto_adjust=True)
        open_prices = pd.DataFrame({ticker: self.downloaded_data[ticker]['Open'] for ticker in self.tickers})
        self.df=open_prices
        if save_files==True :
            open_prices.to_csv("LSTNet\data\large_portfolio.csv")
            np.savetxt("LSTNet\data\large_portfolio.txt", np.array(open_prices), delimiter=',')



class LSTNetModel(FetchStocks):
    def __init__ (self,
                data_path="data\large_portfolio.txt",
                horizon=1,
                save_name="large_portfolio",
                window=7,
                validpercent=0.40,
                batchsize=16,
                skip=7,
                epochs=100,
                cnn_kernel=12,
                learning_rate=0.001,
                dropout=0.3,
                highway=7,
                GRUUnits=100,
                SkipGRUUnits=5,
    ):
        super().__init__()
        self.data_path=data_path
        self.horizon=horizon
        self.save_name=save_name
        self.window=window
        self.validpercent=validpercent
        self.batchsize=batchsize
        self.skip=skip
        self.epochs=epochs
        self.cnn_kernel=cnn_kernel
        self.learning_rate=learning_rate
        self.dropout=dropout
        self.highway=highway
        self.GRUUnits=GRUUnits
        self.SkipGRUUnits=SkipGRUUnits
    
    def train_model(self):
        curr_file_path = os.path.dirname(os.path.abspath(__file__))
        print("training...")
        os.chdir(curr_file_path + "/LSTNet")
        command = f'python main.py --data={curr_file_path + "/LSTNet/data/large_portfolio.txt"} --horizon={self.horizon} --save="save/{self.save_name}_horizon{self.horizon}_window{self.window}_skip{self.skip}" --test --savehistory --logfilename="log/lstnet" --debuglevel=20 --predict="all" --plot --save-plot="save/{self.save_name}_horizon{self.horizon}_window{self.window}_skip{self.skip}_plots" --window={self.window} --validpercent={self.validpercent} --batchsize={self.batchsize} --skip={self.skip} --epochs={self.epochs} --CNNKernel={self.cnn_kernel} --lr={self.learning_rate} --dropout={self.dropout} --highway={self.highway} --GRUUnits={self.GRUUnits} --SkipGRUUnits={self.SkipGRUUnits}' 
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()

        if process.returncode == 0:
            print(output.decode())
        else:
            print("Error:", error.decode())
        print("model saved")
        print(f'Outputs can be found at: LSTNet/save/{self.save_name}_horizon{self.horizon}_window{self.window}_skip{self.skip}')
    
    def get_trained_model(self):
        curr_file_path = os.path.dirname(os.path.abspath(__file__))
        
        custom_objects = {"PreSkipTrans": PreSkipTrans,
                  "PostSkipTrans": PostSkipTrans,
                  "PreARTrans": PreARTrans,
                  "PostARTrans": PostARTrans,
                  }

        json_file = open(f'{curr_file_path + "/LSTNet/save"}/{self.save_name}_horizon{self.horizon}_window{self.window}_skip{self.skip}.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        loaded_model = model_from_json(loaded_model_json, custom_objects=custom_objects)
        loaded_model.load_weights(f'{curr_file_path + "/LSTNet/save"}/{self.save_name}_horizon{self.horizon}_window{self.window}_skip{self.skip}.h5')

        model = loaded_model
        return model

class LSTNetIteratedModel(LSTNetModel, FetchStocks):
    def __init__(self,
                 forecast_steps,
                 n_series,
                 csv_path,
                 save_forecast_json=True
                 ):
        super().__init__()
        self.forecast_steps=forecast_steps
        self.timesteps=self.window
        self.series=np.array(pd.read_csv(csv_path))[:,1:]
        self.json=save_forecast_json
 
    def get_forecast(self):
        model=self.get_trained_model()
        series=self.series
        time_steps=self.window
        forecast_steps=self.forecast_steps
        last_batch=series[-time_steps:,:]
        forecast=[]
        for step in range(forecast_steps):
            pred=model.predict(np.array(last_batch, dtype='float32').reshape(1,time_steps,series.shape[1]))
            forecast.append(pred)
            last_batch=np.append(last_batch[1:,:],pred, axis=0)
        forecast=np.array(forecast)
        
        return forecast

    def plot_lstnet_forecast(self, series_index=24):
        model=self.get_trained_model()
        series=self.series
        time_steps=self.window
        forecast=self.get_forecast()
        forecast_df=pd.DataFrame(forecast.reshape(self.forecast_steps,series.shape[1]))
        forecast_df.columns=self.tickers 

        series_df=pd.DataFrame(series)
        series_df.columns=self.tickers
        cumulative_df=pd.concat([series_df,forecast_df], axis=0)
        cumulative_df=pd.DataFrame(np.array(cumulative_df))
        cumulative_df.iloc[-(100+len(forecast_df)):-len(forecast_df),series_index].plot(color='blue')
        cumulative_df.iloc[-len(forecast_df):,series_index].plot(color='red')
        output_df=cumulative_df.iloc[-55:,:]
    
        if self.json==True:
            json_data = output_df.to_json(orient='index')

            with open('LargePortfolioLSTNet_forecast.json', 'w') as f:
                f.write(json_data)
            print('JSON data saved to', 'LargePortfolioLSTNet_forecast.json')

        plt.legend()
        title_ticker=self.tickers[series_index]
        plt.title(title_ticker)
        plt.show()

