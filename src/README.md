---
# 	LSTNet- Iterated Forecasting
Note: This repository is a followup of the implementation of [fbadine/LSTNet](https://github.com/fbadine/LSTNet) 
The Iterated forecasting method is described in [this](https://arxiv.org/pdf/2003.05672.pdf) paper. This approach is applied here to real-time stocks.
The default arguments of the LSTNet model have been changed a bit, you can see them in the ```run_forecast.py``` file

Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks
[paper](https://arxiv.org/pdf/1703.07015v3.pdf)

## HOW TO USE
``` python
from run_forecast import FetchStocks, LSTNetModel, LSTNetIteratedModel
fetcher=FetchStocks()
fetcher.df
```
Output:
Dataframe of stocks..
``` python
## Creating an instance of the class LSTNetIterated
saved_csv_path=fetcher.csv_path
lstnet_iterated=LSTNetIteratedModel(n_series=48,forecast_steps=15, csv_path=saved_csv_path)
```
    
``` python
lstnet_iterated.train_model()
```

    training...
    Epoch 1/100

     1/38 [..............................] - ETA: 3:14 - loss: 0.5679 - rse: 3.1090 - corr: 0.4719
    10/38 [======>.......................] - ETA: 0s - loss: 0.3846 - rse: 2.4926 - corr: 0.2892  
    20/38 [==============>...............] - ETA: 0s - loss: 0.2997 - rse: 2.0112 - corr: 0.3071
    29/38 [=====================>........] - ETA: 0s - loss: 0.2537 - rse: 1.7078 - corr: 0.3361
    38/38 [==============================] - ETA: 0s - loss: 0.2246 - rse: 1.4998 - corr: nan   
    38/38 [==============================] - 6s 31ms/step - loss: 0.2246 - rse: 1.4998 - corr: nan - val_loss: 0.1273 - val_rse: 1.0666 - val_corr: 0.4246
    .
    .
    .
   
    Epoch 99/100

     1/38 [..............................] - ETA: 0s - loss: 0.0115 - rse: 0.0863 - corr: 0.9857
    10/38 [======>.......................] - ETA: 0s - loss: 0.0120 - rse: 0.0912 - corr: 0.9842
    20/38 [==============>...............] - ETA: 0s - loss: 0.0116 - rse: 0.0876 - corr: 0.9846
    29/38 [=====================>........] - ETA: 0s - loss: 0.0115 - rse: 0.0871 - corr: 0.9846
    38/38 [==============================] - 0s 8ms/step - loss: 0.0114 - rse: 0.0860 - corr: nan - val_loss: 0.0144 - val_rse: 0.1366 - val_corr: 0.5741
    Epoch 100/100

     1/38 [..............................] - ETA: 0s - loss: 0.0123 - rse: 0.0929 - corr: 0.9861
    10/38 [======>.......................] - ETA: 0s - loss: 0.0112 - rse: 0.0848 - corr: 0.9870
    19/38 [==============>...............] - ETA: 0s - loss: 0.0110 - rse: 0.0843 - corr: 0.9873
    29/38 [=====================>........] - ETA: 0s - loss: 0.0110 - rse: 0.0835 - corr: 0.9868
    38/38 [==============================] - 0s 8ms/step - loss: 0.0112 - rse: 0.0850 - corr: nan - val_loss: 0.0144 - val_rse: 0.1360 - val_corr: 0.5748
    training now...
    Figure(1600x1000)
    model saved...

    model saved
    Outputs can be found at: LSTNet/save/large_portfolio_horizon1_window7_skip7
Output: 
Forecast..
``` python
forecast=lstnet_iterated.get_forecast()
```


``` python
lstnet_iterated.plot_lstnet_forecast(series_index=24)
```
![csco](https://github.com/Enforcer03/LSTNet/assets/103068685/4b0ad462-526b-46d5-b4ba-51cb8b59dddd)



