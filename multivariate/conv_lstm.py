import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

def split_df(df, window_size, split=0.75):
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np)-window_size):
        row = [[a] for a in df_as_np[i:i+window_size]]
        X.append(row)
        label = df_as_np[i+window_size]
        y.append(label)
    X=np.array(X)
    y=np.array(y)
    X=X.reshape(-1,time_steps,2)
    X_train=X[:int(split*len(X))]
    y_train=y[:int(split*len(X))]
    X_test=X[int(split*len(X)):]
    y_test=y[int(split*len(X)):]

    return X_train,y_train, X_test, y_test


class ConvLSTM(Model):
    def __init__(self, 
                 time_steps, 
                 n_series,
                 forecast_steps,
                 epochs=20,
                 ):
        super(ConvLSTM, self).__init__()
        self.conv1d = Conv1D(32, kernel_size=2)
        self.lstm = LSTM(64)
        self.dense1 = Dense(32, activation='relu')
        self.dense2 = Dense(16, activation='relu')
        self.dense3 = Dense(n_series, activation='linear')
        self.history= None
        self.forecast_setps=forecast_steps
        self.y_test=None
        self.n_series=n_series
        self.epochs=epochs
        self.time_steps=time_steps
        

    def call(self, inputs):
        x = self.conv1d(inputs)
        x = self.lstm(x)
        x = self.dense1(x)
        x = self.dense2(x)
        outputs = self.dense3(x)
        return outputs

    def train(self, X_train, y_train, X_test, y_test):
        self.y_test=y_test
        cp1 = tf.keras.callbacks.ModelCheckpoint('conv_lstm_checkpoint/', save_best_only=True)
        self.compile(loss=tf.keras.losses.MeanSquaredError(), 
                     optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
                     metrics=[tf.keras.metrics.RootMeanSquaredError()],
                     )
        self.history = self.fit(X_train, 
                                y_train, 
                                validation_data=(X_test, y_test), 
                                epochs=self.epochs,
                                callbacks=[cp1])
    
    def prediction_summary(self, X_test, summary=False):
        preds = self.predict(X_test)
        if summary==True:
            plt.plot(preds[:100, 0], label='preds')
            plt.plot(self.y_test[:100, 0], label='actual')
            plt.legend()
            plt.show()

            plt.plot(self.history.history['loss'])
            plt.plot(self.history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper right')
            plt.show()
        return preds,self.y_test

    def forecast(self, X_test, forecast_steps, plot_forecast=True):
        last_batch = X_test[-self.time_steps:]
        forecast = []
        for step in range(forecast_steps):
            pred = self.predict(np.array(last_batch[step]).reshape(1, time_steps, n_series))
            forecast.append(pred)
            last_batch = np.append(last_batch[1:], pred)
        forecast = np.array(forecast)
        final_forecast = forecast.reshape(forecast_steps, n_series)
        if(plot_forecast==True):
            chart_df=pd.DataFrame(np.append(self.y_test[-100:], final_forecast).reshape(100+forecast_steps,-1))
            chart_df.iloc[:100,1].plot(label='actual')
            chart_df.iloc[100:,1].plot(label='forecast')
            plt.legend()
            plt.show()
            return chart_df
        return final_forecast

time_steps = 7
n_series = 2

inputs = Input(shape=(time_steps, n_series))
conv_lstm_model = ConvLSTM(time_steps, n_series)
outputs = conv_lstm_model(inputs)

conv_lstm_model.summary()

## train here


## forecast
# final_forecast = conv_lstm_model.forecast(X_test, forecast_steps)
# print(final_forecast)

        
        
        