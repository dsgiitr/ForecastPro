import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import pandas as pd;
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import RNN, SimpleRNN
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.layers.core import Activation
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import statsmodels.api as sm

class univariate_ts_rnn:
    def __init__(self,
                 time_steps=10,
                hidden_layers=14,
                epochs=50,
                loss_stopping_patience=20,
                batch_size=15,
                verbose=2,
                loss_curve=True,
                forecast_eval=True,
                activation=['tanh', 'tanh'],
                split=0.75,
                normalize=True,
                learning_rate=0.001,
                return_predictions=True
                ):
        self.time_steps=time_steps
        self.hidden_layers=hidden_layers
        self.epochs=epochs
        self.loss_stopping_patience=loss_stopping_patience
        self.batch_size=batch_size
        self.verbose=verbose
        self.loss_curve=loss_curve
        self.forecast_eval=forecast_eval
        self.split=split
        self.activation=activation
        self.normalize=normalize
        self.learning_rate=learning_rate
        self.return_predictions=return_predictions
    
    def create_RNN(hidden_units, dense_units, input_shape, activation):
        model = Sequential()
        model.add(SimpleRNN(hidden_units, 
                            input_shape=input_shape, 
                            activation=activation[0]))
        model.add(Dense(units=dense_units, 
                        activation=activation[1]))
        model.compile(loss='mean_squared_error', 
                      optimizer='adam',
                      metrics=tf.keras.metrics.MeanAbsolutePercentageError())
        return model

        
    def train(self,data):
        boundry=int(np.floor(len(data)*(self.split)))
        if(self.normalize==True):
            scaler=MinMaxScaler()
            data=scaler.fit_transform(data.reshape(-1,1))
        else:
            data=StandardScaler().fit_transform(data.reshape(-1,1))
       
        train_data=data[:boundry]
        test_data=data[boundry:]

        def get_XY(dat, time_steps):
            Y_ind = np.arange(time_steps, len(dat), time_steps)
            Y = dat[Y_ind]
            rows_x = len(Y)
            X = dat[range(time_steps*rows_x)]
            X = np.reshape(X, (rows_x, time_steps, 1))    
            return X, Y
        time_steps = self.time_steps
        trainX, trainY = get_XY(train_data, time_steps)
        testX, testY = get_XY(test_data, time_steps)

        monitor = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                         min_delta=1e-3, 
                         patience=self.loss_stopping_patience, 
                        verbose=1, 
                        mode='auto', 
                        restore_best_weights=True)

        model = univariate_ts_rnn.create_RNN(hidden_units=self.hidden_layers, 
                        dense_units=1, 
                        input_shape=(time_steps,1), 
                        activation=self.activation
                        )
        history=model.fit(trainX, 
                        trainY,
                        batch_size=self.batch_size, 
                        validation_data=[testX,testY],
                        epochs=self.epochs, 
                        verbose=self.verbose,
                        )
 


        predictions=model.predict(testX).flatten()

        if(self.loss_curve==True):
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper right')
            plt.show()
        
        if(self.forecast_eval==True):
            test_val=test_data[range(0,len(data)-boundry,time_steps)]
            plt.plot(test_val,linewidth=1)
            plt.plot(predictions)
            plt.title('Forecast Evaluation')
            plt.show()
        
            return scaler,model,test_val,predictions
    

class statistical_univariate:
    def __init__(self):
        pass
    def forecast_arima(time_series, order):
        arima_model = sm.tsa.ARIMA(time_series, order=order)
        arima_fit = arima_model.fit()
        arima_forecast = arima_fit.forecast(steps=len(time_series))
        arima_forecast = np.array(arima_forecast[0])
        
        plt.plot(time_series, label='actual')
        plt.plot(arima_forecast, label='forecast')
        plt.legend(loc='best')
        plt.show()
        
        return arima_forecast

