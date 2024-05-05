import torch
from torch import nn
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
        data_df = pd.concat(self.dataframes).sort_index(ascending=True)
        data = self.scaler.fit_transform(data_df.drop(['anomaly'], axis=1))
        sequences = self.create_sequences(data, data_df['anomaly'].to_numpy().astype(float))
        return DataLoader(sequences, batch_size=32, shuffle=True)

    def create_sequences(self, data, target):
        X = []
        y = []
        L = len(data)
        for i in range(L-self.seq_length):
            train_seq = data[i:i+self.seq_length]
            train_label = target[i+1:i+self.seq_length+1]
            X.append(train_seq)
            y.append(train_label)
        return TensorDataset(torch.Tensor(X), torch.Tensor(y))

# Step 3: Define the LSTM model
class LSTM_Model(nn.Module):
    def __init__(self):
        super(LSTM_Model, self).__init__()
        self.lstm = nn.LSTM(input_size=5, hidden_size=64, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)


    def forward(self, input_seq):
        x, _ = self.lstm(input_seq)
        x=self.dropout(x)
        x = x[:,-1]
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Load and process the data


folder_path = './data'
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
dataframes = [pd.read_csv(os.path.join(folder_path, f), index_col=0) for f in csv_files]
seq_length = 5
data_loader = DataLoad(dataframes, seq_length)
data = data_loader.load_and_process_data()


model = LSTM_Model()

loss_fn = BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=5e-5)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)

n_epochs = 100
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in data:
        y_pred = model(X_batch)
        y_pred = torch.sigmoid(y_pred)
        y_batch = y_batch[:, -1].reshape(-1, 1)
        actual_class = torch.round(y_pred)
        data_loader.dataframes[0]['actual_class'] = actual_class

        loss = loss_fn(y_pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    if epoch % 100 != 0:
        continue
    model.eval()
    with torch.no_grad():
        y_pred = model(X_batch)
        train_rmse = np.sqrt(loss_fn(y_pred, y_batch))

    #val_loss = np.mean(val_losses)
    #scheduler.step(val_loss)
    
        print("Epoch %d: train RMSE %.4f" % (epoch, train_rmse))


    """ test function for data prep
    from copy import deepcopy as dc

    def prepare_dataframe_for_lstm(df, n_steps):
        df = dc(df)
        df.set_index('timestamp', inplace=True)
        for i in range(1, n_steps+1):
            df[f'voltage(t-{i})'] = df['voltage'].shift(i)
            df[f'amperage(t-{i})'] = df['amperage'].shift(i)
            df[f'x_pos(t-{i})'] = df['x_pos'].shift(i)
            df[f'y_pos(t-{i})'] = df['y_pos'].shift(i)
            df[f'x_vel(t-{i})'] = df['x_vel'].shift(i)
        df.dropna(inplace=True)
        return df

    lookback = 7
    shifted_df = prepare_dataframe_for_lstm(data_df, lookback)
    shifted_df_as_np = shifted_df.to_numpy()

    shifted_df_as_np
    
    
    """