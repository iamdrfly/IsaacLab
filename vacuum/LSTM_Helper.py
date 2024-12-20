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
# MODEL_PATH = pwd + "/vacuum/model/mass5-20-best/lstm.jit"
# SCALER_FILE = pwd + "/vacuum/model/mass5-20-best/RobustScaler.save"
MODEL_PATH = "/home/amosca/IsaacLab/vacuum/model/mass5-20-best/lstm.jit"
SCALER_FILE = "/home/amosca/IsaacLab/vacuum/model/mass5-20-best/RobustScaler.save"

class TorchRobustScaler:
    def __init__(self, center, scale, device=None):
        self.center = torch.tensor(center, device=device, dtype=torch.float32)
        self.scale = torch.tensor(scale, device=device, dtype=torch.float32)

    def transform(self, tensor):
        """Applica la trasformazione: (x - mediana) / IQR"""
        return (tensor - self.center) / self.scale

    def inverse_transform(self, tensor):
        """Applica la trasformazione inversa: x * IQR + mediana"""
        return tensor * self.scale + self.center

class LSTM_Helper:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.jit.load(MODEL_PATH).to(self.device)
        scaler = joblib.load(SCALER_FILE)
        self.scaler = TorchRobustScaler(center=scaler.center_, scale=scaler.scale_, device=self.device)
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

    def compute_force_tensor(self, press):
        bar = 1.01325
        vacuum_area = 0.000625 * 3  # Calcolo su singolo dito, usa 9 per tutto il piede
        deltap = bar - press  # Operazione su tutto il tensore
        deltap_pascal = deltap * 1e5  # Conversione in Pascal
        forza = deltap_pascal * vacuum_area  # Calcolo della forza
        return forza

    def predict(self, time, voltage, buffer_mode=True):
        ## basato su evaluate_model_unseen() di sweep_lstm.ipynb --> prepara dei dati grezzi e li passa al modello
        if buffer_mode:
            if self.buffer is None:
                self.buffer = torch.zeros((time.shape[0] * time.shape[1], self.seq_length, 2), device=self.device)

            time_voltages = torch.stack((time.flatten(), voltage.flatten()), dim=1)
            X = self.scaler.transform(time_voltages)  # Scaling

            self.buffer[:, :-1, :] = self.buffer[:, 1:, :]  # Shift
            self.buffer[:, -1, :] = X.clone() # Inserimento diretto

            self.model.eval()
            with torch.no_grad():
                pressions = self.model(self.buffer).flatten()

            forces = self.compute_force_tensor(pressions).view(time.shape)

        else:
            v = (voltage+1)/6 # trovo df[V]
            X_df = pd.DataFrame({'time': time, 'voltage': v})  # crea df per scaler
            X_scaled = self.scaler.transform(X_df.values)  # scaling
            # X_scaled = np.stack((np.array(X_df['time']), np.array(X_df['voltage']))).T #fake scaling
            X = self.create_sequences(X_scaled)

            self.model.eval()
            with torch.no_grad():
                input = torch.tensor(X, device=self.device).float()
                pressions = self.model(input).flatten().tolist()

            forces = [self.compute_force(p) for p in pressions]

        return forces
