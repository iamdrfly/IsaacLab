import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import os
import pickle
import json
import datetime
import joblib
import wandb
import os

pwd = os.getcwd()
MODEL_PATH = pwd + "/vacuum/model/mass5-20-best/lstm.jit"
SCALER_FILE = pwd + "/vacuum/model/mass5-20-best/RobustScaler.save"
# MODEL_PATH = "/home/etosin/Documents/IsaacLab_github/vacuum/model/mass5-20-best/lstm.jit"
# SCALER_FILE = "/home/etosin/Documents/IsaacLab_github/vacuum/model/mass5-20-best/RobustScaler.save"

class LSTM_Helper:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.jit.load(MODEL_PATH).to(self.device)
        self.scaler = joblib.load(SCALER_FILE)
        self.seq_length = 6
        self.num_feet = 12
        self.buffer = None

    def create_sequences(self, features):
        sequences = []
        for i in range(len(features) - self.seq_length):
            seq = features[i:i + self.seq_length]
            sequences.append(seq.tolist())
        return sequences

    def compute_force(self, press):
        bar = 1.01325
        vacuum_area = 0.000625 * 3 # calcolo sul singolo dito, usa 9 se vuoi tutto il piede (num_ventose)
        deltap = bar - press
        deltap_pascal = deltap * 10 ** 5
        forza = deltap_pascal * vacuum_area
        return forza

    def predict(self, time, voltage, buffer_mode=True):
        ## basato su evaluate_model_unseen() di sweep_lstm.ipynb --> prepara dei dati grezzi e li passa al modello
        if buffer_mode:
            if self.buffer is None:
                self.buffer = torch.zeros((time.shape[0] * time.shape[1], self.seq_length, 2), device=self.device) # dare shape che sia (num_envs*num_feet, 6, 2)

            time_fl = time.flatten().tolist()
            voltage_fl = voltage.flatten().tolist()
            # v = (voltage+1)/6 # trovo df[V]
            v = voltage_fl
            X = pd.DataFrame({'time': time_fl, 'voltage': v}) # crea df per scaler
            X = self.scaler.transform(X) # scaling

            self.buffer[:, :-1, :] = self.buffer[:, 1:, :] #shift
            for idx, z in enumerate(zip(X[:, 0], X[:, 1])): #assigning new element from the dataset
                self.buffer[idx, -1, 0] = z[0]
                self.buffer[idx, -1, 1] = z[1]

            self.model.eval()
            with torch.no_grad():
                pressions = self.model(self.buffer).flatten().tolist()

            forces_list = [self.compute_force(p) for p in pressions]
            forces = torch.tensor(forces_list).to(self.device).view((time.shape))
        else:
            v = (voltage+1)/6 # trovo df[V]
            X_df = pd.DataFrame({'time': time, 'voltage': v})  # crea df per scaler
            X_scaled = self.scaler.transform(X_df)  # scaling
            # X_scaled = np.stack((np.array(X_df['time']), np.array(X_df['voltage']))).T #fake scaling
            X = self.create_sequences(X_scaled)

            self.model.eval()
            with torch.no_grad():
                input = torch.tensor(X, device=self.device).float()
                pressions = self.model(input).flatten().tolist()

            forces = [self.compute_force(p) for p in pressions]

        return forces
