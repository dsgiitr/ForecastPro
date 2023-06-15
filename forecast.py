def generate_rnn_report(forecast_steps,
                        series,
                          save_json=True,
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
                return_predictions=True):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import tensorflow as tf
    import os
    import pickle
    import json
    from univariate.rnn_univariate import univariate_ts_rnn

    model=univariate_ts_rnn(
        time_steps=time_steps,
                hidden_layers=hidden_layers,
                epochs=epochs,
                loss_stopping_patience=loss_stopping_patience,
                batch_size=batch_size,
                verbose=verbose,
                loss_curve=loss_curve,
                forecast_eval=True,
                activation=activation,
                split=split,
                normalize=True,
                learning_rate=learning_rate,
                return_predictions=True
    )
    scaler,model,past_test_val,past_preds=model.train(data=series)
    past_test_val=np.asarray(past_test_val)
    past_preds=np.asarray(past_preds)

    static_folder = "static"

    if(save_json==True):
        # Create the static folder if it doesn't exist
        if not os.path.exists(static_folder):
            os.makedirs(static_folder)

        # File path for the pickle file
        past_test_json_file_path = os.path.join(static_folder, "past_test.json")
        with open(past_test_json_file_path, "w") as f:
            json.dump(past_test_val.tolist(), f)

        past_preds_json_file_path = os.path.join(static_folder, "past_preds.json")
        with open(past_preds_json_file_path, "w") as f:
            json.dump(past_preds.tolist(), f)

    
    forecast_steps=forecast_steps
    last_batch=series[len(series)-time_steps:]
    forecast=[]
    for step in range(forecast_steps):
        pred=model.predict(np.array(last_batch).reshape(1,10,1))
        forecast.append(pred)
        last_batch=np.append(last_batch[1:],pred)
    forecast=np.array(forecast).flatten()
    final_forecast=scaler.inverse_transform(forecast.reshape(-1,1))
    
    return past_test_val,past_preds,final_forecast
    
        