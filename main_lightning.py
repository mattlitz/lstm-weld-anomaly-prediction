import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

# Step 2: Define the DataLoad class
class DataLoad:
    def __init__(self, folder, seq_length):
        self.folder = folder
        self.seq_length = seq_length
        self.scaler = MinMaxScaler()

    def load_and_process_data(self):
        dataframes = [pd.read_csv(f'{self.folder}/weld_data_00{i}0.csv') for i in range(1, 3)]
        data = pd.concat(dataframes)
        data = self.scaler.fit_transform(data.drop(['timestamp','anomaly'], axis=1))
        sequences = self.create_sequences(data)
        return DataLoader(sequences, batch_size=1, shuffle=True)

    def create_sequences(self, data):
        seq = []
        L = len(data)
        for i in range(L-self.seq_length):
            train_seq = data[i:i+self.seq_length,1:6] #features
            train_label = data[i+self.seq_length, -1]
            seq.append((train_seq ,train_label))
        return TensorDataset(torch.Tensor(seq))

# Step 3: Define the LSTM model
class LSTM(pl.LightningModule):
    def __init__(self, input_size=5, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.loss_function = BCEWithLogitsLoss()

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq.view(len(input_seq) ,1, -1))
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_function(y_pred, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_function(y_pred, y)
        self.log('val_loss', loss, prog_bar=True)

# Load and process the data
folder = './data'  # specify your folder path here
seq_length = 5
data_loader = DataLoad(folder, seq_length)
data = data_loader.load_and_process_data()

# Split data into train and validation sets
train_data = data[:int(len(data)*0.8)]
val_data = data[int(len(data)*0.8):]

# Initialize the model
model = LSTM()

# Initialize the Trainer and fit the model
trainer = pl.Trainer(max_epochs=150, progress_bar_refresh_rate=20, gpus=1)
trainer.fit(model, train_data, val_data)