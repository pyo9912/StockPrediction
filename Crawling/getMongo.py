#############################################
# DB에서 뉴스를 받아와 형태소 분석을 실시      #
# 테스트 후 예측 메인 코드와 연동할 계획       #
#############################################
### DB 연결
from pymongo import MongoClient
import pandas as pd
### 형태소 분석
from konlpy.tag import Kkma, Okt
from datetime import datetime, timedelta
### Word cloud 생성
from gensim.models.ldamodel import LdaModel
from gensim.models.callbacks import CoherenceMetric
from gensim import corpora
from gensim.models.callbacks import PerplexityMetric
from pprint import pprint
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
### 시각화
import pickle
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt


kkma = Kkma()
okt = Okt()

skip_words = ['은','는','이','가','을','를','적','것','들','도','할','수',
                '.',',','\'','\"','와','과','에','에게','로','으로','의',
                '한','하다','했다','있다','이다','입니다','및','[',']',':',
                ';','/','\r','(',')','뉴스','지난','그','득점','시즌','감독',
                '팀','선수','명','등', '결승', '대결']

positive_words = ['수출','수입','증가','성장','급등','급등세','상승','상승세',
                    '기대','기대감','여행','도전','혁신', '유가급락', '유가하락',
                    '기회', '흑자']
negative_words = ['감소','급락','급락세','하락','하락세','코로나','저조','후퇴',
                    '열세','구조조정', '유가급등', '유가상승','위기', '적자',
                    '리스크']

### 문제점 ###
# 최근 항공주 약세는 우크라이나 전쟁 리스크로 인한 국제유가 상승세 때문이다. #
# 위의 문장의 경우 부정문이지만 명사 기준으로 분할하면 긍정문으로 해석할 여지가 있다. #

skip_title = ['배구', '우승', '결승', 'vs', '연승', '연패', '승', '패',
                '선수', '리시브', '스파이크', '서브', '이적', '이적료', 
                '경기', '리그', '블로킹', '세트', '스코어', '점수']

skip_publisher = ['스포']


client = MongoClient(host='localhost', port=27017)

db = client['news_db']

collection = db.news_info

################## 지정 날짜의 collection 내의 문서 불러오기 ###################

# for col in collection.find({"date": "2022.01.15."}):
#     print("날짜: "+col["date"]+"\n"+"글 제목: "+col["title"])

def simpleScore(date_info):
    positive_score = 0
    negative_score = 0
    date = date_info.replace('-','.') + '.'
    for col in collection.find({"date": date}):
        # print("날짜: "+col["date"]+"\n"+"글 제목: "+col["title"])
        news_publisher = col["publisher"]
        news_title = col["title"]
        news_content = col["content"]

        # 스포츠 뉴스사 제외
        isSports = False
        for x in skip_publisher:
            if x in news_publisher:
                # print("스포츠 뉴스사 제외")
                isSports = True
                break
        if isSports: continue

        isSports = False
        for x in skip_title:
            if x in news_title:
                # print("스포츠 뉴스 제외")
                isSports = True
                break
        if isSports: continue

        words = okt.nouns(news_content)
        words = [w for w in words if not w in skip_words]
        positive = [w for w in words if w in positive_words]
        negative = [w for w in words if w in negative_words]
        positive_score = positive_score + len(positive)
        negative_score = negative_score + len(negative)
    # print("Positive Score: ", positive_score)
    # print("Negative Score: ", negative_score)
    return positive_score - negative_score

# score = simpleScore("2022-01-15")
# print(score)


#################### 지정 날짜의 word cloud 생성하기 ##########################

def makeWordCloud(start, end):
    key_words = []
    start_date = datetime.strptime(start, "%Y-%m-%d")
    end_date = datetime.strptime(end, "%Y-%m-%d")
    while start_date <= end_date:
        date = start_date.strftime("%Y.%m.%d.")
        for col in collection.find({"date": date}):
            # print("날짜: "+col["date"]+"\n"+"글 제목: "+col["title"])
            news_date = col["date"]
            news_publisher = col["publisher"]
            news_title = col["title"]
            news_content = col["content"]
            news_url = col["url"]

            # 스포츠 뉴스사 제외
            isSports = False
            for x in skip_publisher:
                if x in news_publisher:
                    # print("스포츠 뉴스사 제외")
                    isSports = True
                    break
            if isSports: continue

            isSports = False
            for x in skip_title:
                if x in news_title:
                    # print("스포츠 뉴스 제외")
                    isSports = True
                    break
            if isSports: continue
            ## 명사 추출
            words = okt.nouns(news_content)
            words = [w for w in words if not w in skip_words]
            ## 뉴스별 빈도수가 많은 상위 10개 명사 추출
            word_list = pd.Series(words)
            top_10_list = word_list.value_counts().iloc[:10]
            key_words.append(list(dict(top_10_list).keys()))
        ## 날짜 이동
        start_date += timedelta(days=1)
    ## 입력 기간동안 받은 명사들로 딕셔너리 생성
    dictionary = corpora.Dictionary(key_words)
    dictionary.filter_extremes(no_below=2, no_above=0.5)
    corpus = [dictionary.doc2bow(text) for text in key_words]
    
    ## 모델링
    num_topics = 5
    chunksize = 2000
    passes = 20
    iterations = 400
    eval_every = None

    temp = dictionary[0]
    id2word = dictionary.id2token

    model = LdaModel(
        corpus = corpus,
        id2word=id2word,
        chunksize=chunksize,
        alpha='auto',
        eta='auto',
        iterations=iterations,
        num_topics=num_topics,
        passes=passes,
        eval_every=eval_every
    )
    
    ## Coherence 확인
    top_topics = model.top_topics(corpus)
    avg_topic_coherence = sum([t[1] for t in top_topics])/num_topics
    print('Average topic coherence: %.4f.'%avg_topic_coherence)

    pprint(top_topics)

    ## 시각화
    lda_visualization = gensimvis.prepare(model, corpus, dictionary, sort_topics=False)
    pyLDAvis.save_html(lda_visualization, 'test_220101_220110.html')
    


makeWordCloud("2022-01-01","2022-01-10")
