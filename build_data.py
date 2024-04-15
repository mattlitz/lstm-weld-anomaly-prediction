import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

# Step 2: Load and process the data
class DataLoad:
    def __init__(self, folder, seq_length):
        self.folder = folder
        self.seq_length = seq_length

    def create_sequences(self, data):
        xs = []
        ys = []
        for i in range(len(data)-self.seq_length-1):
            x = data[i:(i+self.seq_length), 1:5]  # all columns except the last one
            y = data[i+self.seq_length, -1]  # the last column
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def load_and_process_data(self):
        data = []
        for dirpath, dirnames, filenames in os.walk(self.folder):
            for filename in filenames:
                if filename.endswith('.csv'):
                    df = pd.read_csv(os.path.join(dirpath, filename))
                    scaler = MinMaxScaler()
                    df.iloc[:, 1:5] = scaler.fit_transform(df.iloc[:, 1:5])  # normalize X data only
                    x, y = self.create_sequences(df.values)
                    data.append((x, y))
        return data

