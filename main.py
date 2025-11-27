import os
import sys
from utils.parser import parse_args, dir_init
from data_process.data_cleaning import remove_news
from data_process.data_processing import extract_keyword


if __name__ == "__main__":

    args = parse_args()
    args = dir_init(args)

    if args.task == "crawling":
        pass
    elif args.task == "data":
        remove_news(args) # Remove useless news
        extract_keyword(args) # Do sentiment analysis
    elif args.task == "predict":
        pass