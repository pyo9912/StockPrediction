import requests
from bs4 import BeautifulSoup, element
import chardet
from datetime import datetime, timedelta

headers = {'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36'}

def start_crawl(start_date, end_date, db):

    start_date_nospace = start_date.replace('.', '')
    end_date_nopspace = end_date.replace('.', '')
    current_page = 1

    searching = True
    search_url = f"https://search.naver.com/search.naver?where=news&sm=tab_pge&query=%EB%8C%80%ED%95%9C%ED%95%AD%EA%B3%B5&sort=2&photo=0&field=0&pd=3&ds={start_date}&de={end_date}&mynews=0&office_type=0&office_section_code=0&news_office_checked=&nso=so:r,p:from{start_date_nospace}to{end_date_nopspace},a:all&start=1"
    while searching:
        naver_response = requests.get(search_url, headers=headers)
        naver_html_src = naver_response.text
        naver_soup = BeautifulSoup(naver_html_src, 'html.parser')
        pages = naver_soup.find('div', {'class' : 'sc_page_inner'})
        print(search_url)
        print(current_page)
        # 뉴스 리스트 
        news_list = naver_soup.select('li[class="bx"]')
        
        for news in news_list:
            # 뉴스 제목
            news_title = news.find('a', attrs={'class':'news_tit'}).get('title')
            #print(news_title)
            # 뉴스 원본 링크 
            news_url = news.find('a', attrs={'class':'news_tit'}).get('href')
            print(news_url)
            # 뉴스 날짜 
            find_date = news.find_all('span', attrs={'class':'info'})
            if len(find_date) > 1:
                news_date = find_date[1].text
            else:
                news_date = find_date[0].text
            print(news_date)
            # 뉴스 발행사 
            try:
                news_publisher = news.find('a', attrs={'class':'info press'}).text
            except:
                try:
                    news_publisher = news.find('a', attrs={'class':'info'}).text
                except:
                    continue
            print(news_publisher)
            #print(news_publisher)
            # 뉴스 내용
            news_summary = news.find('a', attrs={'class':'api_txt_lines dsc_txt_wrap'}).text
            news_summary = news_summary[0:40]
            #print(news_summary)

            if "일" in news_date:
                news_date = datetime.today() - timedelta(days=int(news_date[0]))
                news_date = news_date.strftime("%Y.%m.%d")
            #print(news_date)
            try:
                news_response = requests.get(news_url, headers=headers)
            except:
                db.news_info.update_one({"title":news_title, "publisher":news_publisher, "date":news_date}, {'$set':{ "url":news_url, "content":""}}, upsert=True)
                continue

            if news_summary in str(news_response.text):
                news_soup = BeautifulSoup(news_response.text, 'html.parser')
            else:
                news_soup = BeautifulSoup(news_response.content.decode('euc-kr', 'replace'), 'html.parser')

            news_content = str(news_soup.text).split('\n')

            news_index = [i for i, s in enumerate(news_content) if news_summary in s]
            if news_index:
                news_index = news_index[0]

                news_text = []
                punctuations = ['.', ',', '?', '!']
                count = 0
                if news_index != 0:
                    for i in reversed(range(0, news_index)):
                        if any(text in news_content[i] for text in punctuations): 
                                news_text.append(news_content[i])
                                count = 0
                                #print("i", i, "count", count)
                                #print(news_content[i])
                        else:
                            count +=1
                        if count > 3:
                            break
                
                if news_text:
                    news_text.reverse()
                count = 0

                news_text.append(news_content[news_index])
                for i in range(news_index, len(news_content)):
                    
                    if any(text in news_content[i] for text in punctuations): 
                        news_text.append(news_content[i])
                        count = 0

                    else:
                        count +=1
                    if count > 3:
                        break
                
                news_text = ' '.join(news_text)
                #print(news_text)

                db.news_info.update_one({"title":news_title, "publisher":news_publisher, "date":news_date}, {'$set':{ "url":news_url, "content":news_text}}, upsert=True)


        try:
            current_page = current_page + 1
            next_page_url = [p for p in pages.find_all('a') if p.text == str(current_page)][0].get('href')
            next_page_num = [p for p in pages.find_all('a') if p.text == str(current_page)][0].text
            search_url = "https://search.naver.com/search.naver" + next_page_url 
            #print(current_page)
        except:
            searching=False
            #print("Page does not exist")

    return news_date, current_page

