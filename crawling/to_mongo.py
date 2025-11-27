from pymongo import MongoClient
from crawl_naver import start_crawl
from datetime import date, timedelta

client = MongoClient(host='localhost', port=27017)

db = client['news_db']

#print(db.list_collection_names)

start_date = date(2020, 6, 11)
end_date = date(2022, 5, 1)

while start_date <= end_date:
    start_date_string = start_date.strftime("%Y.%m.%d")
    last_date = start_date+timedelta(3)
    last_date_string = last_date.strftime("%Y.%m.%d")
    
    print(start_date_string, last_date_string)
    finish_date, last_page = start_crawl(start_date_string, last_date_string, db)

    if finish_date != last_date_string and last_page == 400:
        start_date = finish_date.strptime("%Y.%m.%d")
    else:
        start_date = last_date

    