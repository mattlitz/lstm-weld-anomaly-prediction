import torch
from torch import nn
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import os

# Step 2: Define the DataLoad class
class DataLoad:
    def __init__(self, dataframes, seq_length):
        self.dataframes = dataframes
        self.seq_length = seq_length
        self.scaler = MinMaxScaler()

    def load_and_process_data(self):
        data = pd.concat(self.dataframes)
        data = self.scaler.fit_transform(data.drop(['timestamp','anomaly'], axis=1))
        sequences = self.create_sequences(data.drop(['timestamp','anomaly'], axis=1))
        return DataLoader(sequences, batch_size=1, shuffle=True)

    def create_sequences(self, data):
        seq = []
        L = len(data)
        for i in range(L-self.seq_length):
            train_seq = data[i:i+self.seq_length]
            train_label = data[i+self.seq_length, -1]
            seq.append((train_seq ,train_label))
        return TensorDataset(torch.Tensor(seq))

# Step 3: Define the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size=5, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq.view(len(input_seq) ,1, -1))
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

# Load and process the data


folder_path = './data'
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
dataframes = [pd.read_csv(os.path.join(folder_path, f)) for f in csv_files]
seq_length = 5
data_loader = DataLoad(dataframes, seq_length)
data = data_loader.load_and_process_data()



# Initialize the model
model = LSTM()

# Initialize the loss function and optimizer
loss_function = BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Define the training loop
for epoch in range(150):
    for seq, labels in data:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()