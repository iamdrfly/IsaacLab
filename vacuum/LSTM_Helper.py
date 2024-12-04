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

class LSTM_Helper:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.jit.load(MODEL_PATH).to(self.device)
        self.scaler = joblib.load(SCALER_FILE)
        self.seq_length = 6

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

    def predict(self, time, voltage):
        ## basato su evaluate_model_unseen() di sweep_lstm.ipynb --> prepara dei dati grezzi e li passa al modello

        # v = (voltage+1)/6 # trovo df[V]
        v = voltage
        X = pd.DataFrame({'time': time, 'voltage': v}) # crea df per scaler
        X = self.scaler.transform(X) # scaling
        X = self.create_sequences(X)

        self.model.eval()
        with torch.no_grad():
            pressions = self.model(torch.tensor(X, device=self.device).float()).flatten().tolist()

        forces = [self.compute_force(p) for p in pressions]

        return forces
