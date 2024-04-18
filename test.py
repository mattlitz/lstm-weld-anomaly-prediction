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
        data_df = pd.concat(self.dataframes)
        data = self.scaler.fit_transform(data_df.drop(['timestamp','anomaly'], axis=1))
        sequences = self.create_sequences(data, data_df['anomaly'].to_numpy().astype(float))
        return DataLoader(sequences, batch_size=1, shuffle=True)

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
class LSTM(nn.Module):
    def __init__(self, input_size=5, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        out, _ = self.lstm(input_seq)
        out = out[:,-1]
        pred = self.linear(out)
        return pred.squeeze()

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
loss_fn = BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=0.001)

n_epochs = 200
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in data:
        y_pred = model(X_batch)
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