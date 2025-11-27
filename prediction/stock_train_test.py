import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime


#############################################
# PyTorch 모델 정의
#############################################
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(1, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True)
        self.fc1 = nn.Linear(32, 25)
        self.fc2 = nn.Linear(25, 1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

#############################################
# 모델 학습
#############################################
def train_lstm(model, x_train, y_train, epochs=100, lr=0.001):
    crit = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(x_train)
        loss = crit(pred, y_train)
        loss.backward()
        opt.step()
        if (ep + 1) % 10 == 0:
            print(f"Epoch {ep+1}/{epochs} loss: {loss.item():.6f}")

    return model

#############################################
# 예측 + 시각화
#############################################
def test_lstm(model, x_test, y_test, sent_test, scaler, base_df):
    model.eval()
    preds = model(x_test).detach().numpy()
    preds = scaler.inverse_transform(preds)

    preds += sent_test * 100

    real = base_df['종가'].values[len(base_df)//10*7:]

    plt.figure(figsize=(16, 8))
    plt.plot(real, label='Real')
    plt.plot(preds, label='Prediction')
    plt.legend()
    plt.show()

    return preds
