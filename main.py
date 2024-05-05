########################
#
# LSTM Weld Anomaly Prediction
#
# author: @mattlitz
#%% 
import os
import numpy as np
import pandas as pd
import csv

import warnings
warnings.filterwarnings('ignore')

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from build_data import DataLoad

from tqdm import tqdm

#%%

class LSTM(nn.Module):
    def __init__(self, input_size=5, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


#%% 

if __name__ == '__main__':
    
    # Define the training loop
    def train(model, data, loss_function, optimizer, epochs):
        writer = SummaryWriter()
        for i in range(epochs):
            for x, y in tqdm(data, desc='Training epoch {}'.format(i+1)):
                x = torch.tensor(x, dtype=torch.float32)
                y = torch.tensor(y, dtype=torch.float32)
                optimizer.zero_grad()
                model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                torch.zeros(1, 1, model.hidden_layer_size))

                y_pred = model(x)

                single_loss = loss_function(y_pred, y)
                single_loss.backward()
                optimizer.step()
            writer.add_scalar('Loss/train', single_loss, i)
        writer.close()

    # Evaluate the model
    def evaluate(model, data, loss_function):
        model.eval()
        total_loss = 0
        for x, y in data:
            x = torch.tensor(x, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)
            with torch.no_grad():
                y_pred = model(x)
                single_loss = loss_function(y_pred, y)
                total_loss += single_loss.item()
        return total_loss / len(data)

    model = LSTM()
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    # Load and process the data
    folder = './data'
    seq_length = 5
    data_loader = DataLoad(folder, seq_length)
    data = data_loader.load_and_process_data()

    train(model, data, loss_function, optimizer, epochs=150)
    loss = evaluate(model, data, loss_function)
    print(f'Test loss: {loss}')

    # Step 7: Create new dataframes of original data with target prediction results
    for dirpath, dirnames, filenames in os.walk(folder):
        for filename in filenames:
            if filename.endswith('.csv'):
                df = pd.read_csv(os.path.join(dirpath, filename))
                x, y = DataLoad.create_sequences(df.values, seq_length)
                x = torch.tensor(x, dtype=torch.float32)
                with torch.no_grad():
                    predictions = model(x)
                predictions = predictions.numpy()
                df['Predictions'] = np.append([np.nan]*seq_length, predictions)
                df.to_csv(os.path.join(dirpath, 'predicted_' + filename), index=False)



# %%
