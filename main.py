import os
import sys
import pandas as pd
from utils.parser import parse_args, dir_init
from utils.utils import preprocess, create_dataset
from data_process.data_cleaning import remove_news
from data_process.data_processing import extract_keyword
from prediction.stock_train_test import LSTMModel, train_lstm, test_lstm

if __name__ == "__main__":

    args = parse_args()
    args = dir_init(args)

    if args.task == "crawling":
        pass
    elif args.task == "data":
        remove_news(args) # Remove useless news
        extract_keyword(args) # Do sentiment analysis
    elif args.task == "predict":
        ### Fix train/test data ###
        news_paths = [
            os.path.join(args.home,'data/result','2020_result.json'),
            os.path.join(args.home,'data/result','2021_result.json'),
            os.path.join(args.home,'data/result','2022_result.json'),
        ]
        df_list = [pd.read_json(p) for p in news_paths]
        news_df = pd.concat(df_list)
        stock_path = os.path.join(args.home,'data/news_data_update','stock.csv')
        stock_df = pd.read_csv(stock_path, encoding='euc-kr')
        base_df = preprocess(news_df, stock_df)

        x_train, y_train, x_test, y_test, sent_test, scaler = create_dataset(base_df)

        model = LSTMModel()
        model = train_lstm(model, x_train, y_train, epochs=100)
        test_lstm(model, x_test, y_test, sent_test, scaler, base_df)