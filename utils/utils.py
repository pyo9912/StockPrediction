import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler


#############################################
# 데이터 전처리 함수
#############################################
def preprocess(news_df, stock_df):
    stock_df['date'] = stock_df['일자'].apply(lambda x: datetime.strptime(x, '%Y/%m/%d'))
    stock_df.drop(['일자'], axis=1, inplace=True)

    stock = stock_df.loc[(stock_df['date'] >= '2020-01-02') & (stock_df['date'] <= '2022-09-09')]
    merge = news_df.merge(stock, how='inner', on='date')

    merge['label'] = merge['대비'].apply(lambda x: 0 if x <= 0 else 1)

    # vocab
    vocab = {}
    for keyWord in merge['keyWord']:
        for k in keyWord:
            if k not in vocab and len(k) > 1:
                vocab[k] = 0

    up = len(merge[merge['label'] == 1])
    down = len(merge[merge['label'] == 0])
    up_ratio = up / (up + down)
    down_ratio = down / (up + down)

    for i, keyWord in enumerate(merge['keyWord']):
        for k in keyWord:
            if len(k) >= 2:
                vocab[k] += down_ratio if merge.iloc[i]['label'] == 1 else -up_ratio

    vocab_v = np.array(list(vocab.values())).reshape(-1, 1)
    scaler_vocab = MinMaxScaler()
    vocab_scaled = scaler_vocab.fit_transform(vocab_v).flatten()

    for i, k in enumerate(vocab.keys()):
        vocab[k] = vocab_scaled[i]

    total = []
    for i, keyWord in enumerate(merge['keyWord']):
        score = sum(vocab[k] for k in keyWord if k in vocab)
        total.append(score / len(keyWord))

    merge['sentiment_score'] = total

    base_df = merge[['date', '종가', 'sentiment_score']]
    base_df['sentiment_sum'] = base_df.groupby(base_df.date).sentiment_score.cumsum()
    base_df = base_df.drop_duplicates(['date', '종가'], keep='last')

    return base_df

#############################################
# Train/Test 데이터 생성 함수
#############################################
def create_dataset(base_df, seq=8):
    close_data = base_df['종가'].values.reshape(-1, 1)
    sent_data = base_df['sentiment_sum'].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_data)

    train_size = int(len(close_data) * 0.7)
    train = scaled[:train_size]
    test = scaled[train_size - seq:]
    sent_test = sent_data[train_size:]

    def make_xy(data):
        xs, ys = [], []
        for i in range(seq, len(data)):
            xs.append(data[i-seq:i, 0])
            ys.append(data[i, 0])
        return np.array(xs), np.array(ys)

    x_train, y_train = make_xy(train)
    x_test, y_test = make_xy(test)

    x_train = torch.tensor(x_train).float().unsqueeze(-1)
    y_train = torch.tensor(y_train).float().unsqueeze(-1)

    x_test = torch.tensor(x_test).float().unsqueeze(-1)
    y_test = torch.tensor(y_test).float().unsqueeze(-1)

    return x_train, y_train, x_test, y_test, sent_test, scaler