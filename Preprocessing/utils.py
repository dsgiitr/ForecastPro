import pandas as pd
import numpy as np
import torch

class take_input:
    def __init__(self, file_path, target_variable, return_type='df'):
        self.file_path = file_path
        self.target_variable = target_variable
        self.return_type = return_type
        
    def return_data(self):
        if self.file_path.endswith('.csv'):
            df = pd.read_csv(self.file_path)
        else:
            df = pd.read_excel(self.file_path)
        
        X = df.drop(self.target_variable, axis=1)
        y = df[self.target_variable]
        
        if self.return_type == 'df':
            return X, y
        elif self.return_type == 'np':
            return  np.array(X.values), np.array(y.values)
        elif self.return_type == 'tensor':
            return torch.tensor(X.values, dtype=torch.float32), torch.tensor(y.values, dtype=torch.float32)
