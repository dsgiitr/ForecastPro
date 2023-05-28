import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn')
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
from keras.layers import BatchNormalization
from sklearn.preprocessing import StandardScaler,MinMaxScaler

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
                return_predictions=True,
                recurrent_dropout=0.4
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
        self.recurrent_dropout=recurrent_dropout
    
    def create_RNN(self,hidden_units, dense_units, input_shape, activation):
        model = Sequential()
        model.add(SimpleRNN(hidden_units, 
                            input_shape=input_shape, 
                            activation=activation[0]),
                            )
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
            scaler=StandardScaler()
            data=scaler.fit_transform(data.reshape(-1,1))
       
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

        model = self.create_RNN(hidden_units=self.hidden_layers, 
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
        predictions=scaler.inverse_transform(np.array(predictions).reshape(-1,1))
        test_data=scaler.inverse_transform(np.array(test_data).reshape(-1,1))

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
            plt.plot(test_val,linewidth=1,marker='o')
            plt.plot(predictions,marker='o')
            plt.title('Forecast Evaluation')
            plt.show()
        
            return test_val,predictions



class multivariate_ts_lstm:
    def __init__(self,
                 time_steps=10,
                lstm_hidden_layers=50,
                epochs=50,
                loss_stopping_patience=20,
                n_past_days=56,
                batch_size=7,
                loss_curve=True,
                forecast_eval=True,
                split=0.75,
                learning_rate=0.001,
                return_predictions=True,
                recurrent_dropout=[0.4,0.4]
                 ):
        self.time_steps=time_steps
        self.lstm_hidden_layers=lstm_hidden_layers
        self.epochs=epochs
        self.loss_stopping_patience=loss_stopping_patience
        self.length=n_past_days
        self.batch_size=batch_size
        self.loss_curve=loss_curve
        self.forecast_eval=forecast_eval
        self.split=split
        self.learning_rate=learning_rate
        self.return_predictions=return_predictions
        self.recurrent_dropout=recurrent_dropout

        

    def train_predict(self,data):   
        ## prints model summary, loss curve and returns test, predictions 
        dataset=data
        train_size = int(len(dataset) * (self.split))
        test_size = len(dataset) - train_size
        train, self.test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

        ## preparing data for forecasting
        from keras.preprocessing.sequence import TimeseriesGenerator 
        length = self.length
        batch_size = self.batch_size
        train_generator = TimeseriesGenerator(train,
                                            train,
                                            length=length,
                                            batch_size=batch_size)
        print("Total number of samples in the original training data = ", len(train))
        print("Total number of samples in the generated data = ", len(train_generator))

        validation_generator = TimeseriesGenerator(self.test, 
                                           self.test, 
                                           length=length ,
                                           batch_size=batch_size)

        num_features=dataset.shape[1]

        ## creating model
        model = Sequential()
        model.add(LSTM(self.lstm_hidden_layers, 
                    activation='relu', 
                    return_sequences=True, 
                    input_shape=(length, num_features),
                    recurrent_dropout=self.recurrent_dropout[0])
                    )
        model.add(LSTM(self.lstm_hidden_layers, 
                       activation='relu',
                       recurrent_dropout=self.recurrent_dropout[1])
                       )
        model.add(Dense(32))
        ##model.add(Dropout(rate=0.5))
        model.add(Dense(num_features))
        model.compile(optimizer='adam', 
                    loss='mean_squared_error',
                    metrics=tf.keras.metrics.MeanAbsolutePercentageError(),
                    )

        model.summary()

        ## fitting the model
        monitor = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                         min_delta=1e-3, 
                         patience=self.loss_stopping_patience, 
                        verbose=1, 
                        mode='auto', 
                        restore_best_weights=True)
        history=model.fit_generator(generator=train_generator, 
                            verbose=2, 
                            epochs=self.epochs, 
                            validation_data=validation_generator,
                            callbacks=monitor)
        if(self.loss_curve==True):
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper right')
            plt.show()
        
        self.predictions=model.predict(
        validation_generator
        )

        if(self.return_predictions==True):
            return self.test,self.predictions
        
    def plot_predictions(self,index):
        import matplotlib.pyplot as plt
        temp_df=pd.concat([pd.DataFrame(self.predictions[:,index]),pd.DataFrame(self.test[:len(self.predictions),index])],axis=1)
        temp_df.plot(linewidth=1.7,marker = 'o')
        plt.legend(['forecasted','actual'])
        plt.show()
        

class error_metrics:
    def __init__(self):
        return None
    
    def SMAPE(y_true, y_pred):
        ans = 0
        for i in range(len(y_true)):
            num = np.abs(y_true[i] - y_pred[i])
            denom = (np.abs(y_true[i]) + np.abs(y_pred[i]))/2
            ans = ans + np.sum((np.divide(num,denom)))
            return (100/(len(y_true)*len(y_true[0])))*ans 
    
    def percentage_error(y_true,y_pred):
        from sklearn.metrics import mean_absolute_error
        mae=mean_absolute_error(y_true,y_pred)
        m1=y_true.mean()
        m2=y_pred.mean()
        print('Peercentage error'," ",(mae/(m1+m2))*100, "%")

    


