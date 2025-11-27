import math
import numpy as np
import pandas as pd
from datetime import datetime
import os
import sys
import matplotlib.pyplot as plt
from pprint import pprint

# PyTorch imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils.parser import parse_args
from sklearn.preprocessing import MinMaxScaler, RobustScaler

# reproducibility
torch.manual_seed(42)
np.random.seed(42)

# -------------------------
# 데이터 불러오기 및 전처리
# -------------------------
args = parse_args()
news_paths = [
            os.path.join(args.home,'data/result','2020_result.json'),
            os.path.join(args.home,'data/result','2021_result.json'),
            os.path.join(args.home,'data/result','2022_result.json'),
        ]
df_list = [pd.read_json(p) for p in news_paths]
df = pd.concat(df_list)
stock_path = os.path.join(args.home,'data/news_data_update','stock.csv')
stockdf = pd.read_csv(stock_path, encoding='euc-kr')

stockdf['date'] = stockdf['일자'].apply(lambda x: datetime.strptime(x, '%Y/%m/%d'))
stockdf.drop(['일자'], axis=1, inplace=True)

stock = stockdf.loc[(stockdf['date'] >= '2020-01-02') & (stockdf['date'] <= '2022-09-09')]

merge = df.merge(stock, how='inner', on='date')
merge['label'] = merge['대비'].apply(lambda x: 0 if x <= 0 else 1)

# vocab 생성
vocab = {}
for keyWord in merge['keyWord']:
    for k in keyWord:
        if k in vocab or len(k) <= 1:
            pass
        else:
            vocab[k] = 0

# up/down 비율
up = len(merge[merge['label'] == 1])
down = len(merge[merge['label'] == 0])
up_ratio = up / (up + down)
down_ratio = down / (up + down)

# 단어 점수 산출
for i, keyWord in enumerate(merge['keyWord']):
    for k in keyWord:
        if merge.iloc[i]['label'] == 1:
            if (len(k) >= 2):
                vocab[k] += down_ratio
        else:
            if (len(k) >= 2):
                vocab[k] -= up_ratio

# vocab 정규화 (원본은 RobustScaler 사용)
robustScaler = RobustScaler()
vocab_v = [v for v in vocab.values()]
scaled_list = robustScaler.fit_transform(np.array(vocab_v).reshape(-1,1))
scaled_list = scaled_list.reshape(1,-1)

for i, v in enumerate(vocab_v):
    vocab_v[i] = scaled_list[0][i]

for i, k in enumerate(vocab.keys()):
    vocab[k] = vocab_v[i]

# sentiment score 계산
total = []
for i, keyWord in enumerate(merge['keyWord']):
    sent_score = 0
    for k in keyWord:
        if (len(k) >= 2 and k in vocab):
            sent_score += vocab[k]
    total.append(sent_score/len(keyWord))

merge['sentiment_score'] = total

# median 기준 label
desc = merge['sentiment_score'].describe()
median = desc['50%']
merge['sentiment_label'] = 0
merge.loc[merge.query(f'sentiment_score >= {median}').index, 'sentiment_label'] = 1

base_df = merge[['date', '종가', 'sentiment_score']].copy()
base_df['sentiment_sum'] = base_df.groupby(base_df.date).sentiment_score.cumsum()
base_df = base_df.drop_duplicates(['date','종가'], keep = 'last')

# -------------------------
# 시계열 데이터 구성 (원본과 동일)
# -------------------------
date_df = base_df.filter(['date'])
date_data = date_df.values

close_df = base_df.filter(['종가'])
close_data = close_df.values.astype(np.float32)

sent_df = base_df.filter(['sentiment_sum'])
sent_data = sent_df.values.astype(np.float32)

# scaling (close price)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(close_data)

# train/test split
seq = 8
train_data_len = math.ceil(len(close_data) * 0.7)
train_data = scaled_data[0:train_data_len, :]
test_data = scaled_data[train_data_len-seq:, :]
sent_score = sent_data[train_data_len:, :]

# create train sequences
x_train = []
y_train = []
for i in range(seq, len(train_data)):
    x_train.append(train_data[i-seq:i, 0])
    y_train.append(train_data[i, 0])

x_train = np.array(x_train)
y_train = np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# create test sequences
x_test = []
y_test = close_data[train_data_len:, :].astype(np.float32)
for i in range(seq, len(test_data)):
    x_test.append(test_data[i-seq:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# -------------------------
# PyTorch dataset / dataloader
# -------------------------
class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float().unsqueeze(-1)  # shape (N,1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

batch_size = 32
train_dataset = SequenceDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# -------------------------
# PyTorch LSTM model (두 계층 구조 재현)
# -------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden1=64, hidden2=32, fc1=25):
        super(LSTMModel, self).__init__()
        # 첫 LSTM: return_sequences=True -> batch_first True, num_layers=1
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden1, batch_first=True)
        # 두번째 LSTM: return_sequences=False
        self.lstm2 = nn.LSTM(input_size=hidden1, hidden_size=hidden2, batch_first=True)
        # fully connected
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(hidden2, fc1)
        self.fc2 = nn.Linear(fc1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, seq, feat)
        out1, _ = self.lstm1(x)           # out1: (batch, seq, hidden1)
        out2, _ = self.lstm2(out1)        # out2: (batch, seq, hidden2)
        # take last time-step from out2
        last = out2[:, -1, :]             # (batch, hidden2)
        x = self.fc1(last)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = LSTMModel(input_size=1).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -------------------------
# training loop
# -------------------------
epochs = 500  # 원본과 동일
model.train()
for epoch in range(epochs):
    epoch_loss = 0.0
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * xb.size(0)
    epoch_loss /= len(train_loader.dataset)
    if (epoch + 1) % 50 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.6f}")

# -------------------------
# prediction
# -------------------------
model.eval()
with torch.no_grad():
    X_test = torch.from_numpy(x_test).float().to(device)
    preds = model(X_test).cpu().numpy()  # shape (N,1)

# inverse scale predictions
preds_unscaled = scaler.inverse_transform(preds)  # shape (N,1)

print("preds_unscaled shape:", np.shape(preds_unscaled))
print("sent_score shape:", np.shape(sent_score))

# 감성 점수 더하기 (원래 코드: lstmPredictions += (sent_score * 100))
# Ensure shapes match: sent_score is (N,1)
preds_unscaled = preds_unscaled + (sent_score * 100)

# LSTM RMSE (원본은 sqrt(mean(pred - y_test)**2))
y_test_arr = y_test.astype(np.float32)
rmse = np.sqrt(np.mean((preds_unscaled.flatten() - y_test_arr.flatten())**2))
print("LSTM RMSE: ", rmse)

# -------------------------
# 시각화 (원본과 동일한 플롯)
# -------------------------
train_len = train_data_len

train = close_df[:train_data_len].copy().reset_index(drop=True)
valid = close_df[train_data_len:].copy().reset_index(drop=True)

# Prepare prediction column
valid = valid.iloc[:len(preds_unscaled)].copy()
valid['Predictions'] = preds_unscaled

# plotting
plt.figure(figsize=(16,9))
plt.title('LSTM Model + Sentimental Score')
plt.xlabel("Date", fontsize=18)
plt.ylabel("Price", fontsize=18)

plt.plot(date_df.iloc[0:train_len].values, close_df[:train_data_len].values, label='Train')
plt.plot(date_df.iloc[train_len:].iloc[:len(preds_unscaled)].values, valid[['종가']].values, label='Val')
plt.plot(date_df.iloc[train_len:].iloc[:len(preds_unscaled)].values, valid[['Predictions']].values, label='Predictions')
plt.legend(['Train','Val','Predictions'], loc='lower right')
plt.show()
