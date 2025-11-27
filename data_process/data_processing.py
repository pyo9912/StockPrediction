import os
import sys
import torch
from tqdm import tqdm
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import itertools
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from konlpy.tag import Okt  # ckonlpy 대신 Okt 사용

# ------------------------------
# 텍스트 처리 유틸리티
# ------------------------------
def joinText(textList):
    textList = list(set(textList))
    return ' '.join(textList) if len(textList) > 1 else textList[0]

def getResult(x):
    return x[0]['label']

def most_frequent(idx, df):
    labellist = df.iloc[idx, 3]
    occurence_count = Counter(labellist)
    return occurence_count.most_common(1)[0][0]

def convertToText(x, df):
    return df.iloc[x, 2].tolist()

def convertToDate(x, df):
    return df.iloc[x, 0].date()

# ------------------------------
# 사용자 사전 + 토크나이저 정의
# ------------------------------
class MyTokenizer:
    def __init__(self, stopwords, companyList):
        self.stopwords = set(stopwords)
        self.okt = Okt()
        self.companyList = set(companyList)

    def __call__(self, text):
        # 형태소 분석
        tokens = self.okt.pos(text, stem=True)
        # 명사/동사만 선택 + 길이 2 이상 + 불용어 제거
        words = [word for word, tag in tokens if tag in ['Noun', 'Verb'] and len(word) > 1 and word not in self.stopwords]
        # 회사명이 문장에 포함되어 있다면 무조건 단어 목록에 추가
        for company in self.companyList:
            if company in text:
                words.append(company)
        return words

# ------------------------------
# 키워드 추출
# ------------------------------
def extractKeyword(text, tk):
    vectorizer = CountVectorizer(tokenizer=tk, token_pattern=None, min_df=0.3)
    try:
        X = vectorizer.fit_transform(text)
        idx2vocab = [vocab for vocab, idx in sorted(vectorizer.vocabulary_.items(), key=lambda x:x[1])]
        return idx2vocab
    except ValueError:
        return []

# ------------------------------
# 메인 키워드 추출 함수
# ------------------------------
def extract_keyword(args):
    year = args.year

    # 데이터 불러오기
    file_path = os.path.join(args.home,"data/news_data_update",f"news_{year}_update.json")
    raw_df = pd.read_json(file_path)
    raw_df['keySentence'] = raw_df['content'].apply(lambda s: s[1])
    df = raw_df[['date', 'title', 'keySentence']].copy()
    df = df[df['keySentence'].str.len() != 0].reset_index(drop=True)
    if args.debug:
        df = df.iloc[:100]   #
    print(f'{year} Dataframe Created')

    # KeySentence 합치기
    df['keySentence'] = df['keySentence'].apply(joinText)

    # 코퍼스 준비
    corpus = df['keySentence'].tolist()
    queries = corpus

    # SentenceTransformer 임베딩
    embedder = SentenceTransformer("jhgan/ko-sroberta-multitask")
    batch_size = args.batch_size
    corpus_embeddings = []

    for i in tqdm(range(0, len(corpus), batch_size)):
        batch = corpus[i:i+batch_size]
        emb = embedder.encode(batch, convert_to_tensor=True)
        corpus_embeddings.append(emb)

    # 리스트 → 텐서로 합치기
    corpus_embeddings = torch.cat(corpus_embeddings, dim=0)
    # corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

    df_idx = list(df.index)
    res_idx, res_name = [], []

    while len(df_idx) != 0:
        query = queries[df_idx[0]]
        query_embedding = embedder.encode(query, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0].cpu()

        # 유사도 0.6 이상
        res = [i for i, v in enumerate(cos_scores) if v > 0.6]
        res_name.append(df_idx[0])
        res_idx.append(res)
        df_idx = [x for x in df_idx if x not in res]

    res_df = pd.DataFrame({'date': res_name, 'news': res_idx})
    print(f'{year} similarity groupped')

    # Sentiment 분석
    tokenizer = AutoTokenizer.from_pretrained("snunlp/KR-FinBert-SC")
    model = AutoModelForSequenceClassification.from_pretrained("snunlp/KR-FinBert-SC")
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    df['predicted'] = df['title'].apply(lambda x: classifier(x))
    df['predicted'] = df['predicted'].apply(getResult)
    res_df['sentiment'] = res_df['news'].apply(lambda x: most_frequent(x, df))
    res_df['news'] = res_df['news'].apply(lambda x: convertToText(x, df))
    res_df['date'] = res_df['date'].apply(lambda x: convertToDate(x, df))
    print(f'{year} Title Analysis Completed')

    # 키워드 추출
    stopword_path = os.path.join(args.home,'data/csv_files','stopword.csv')
    stopword_df = pd.read_csv(stopword_path)
    stopwordList = stopword_df['단어'].tolist()
    company_path = os.path.join(args.home,'data/csv_files','company_list.csv')
    company_df = pd.read_csv(company_path)
    companyList = company_df['companyName'].tolist()
    tk = MyTokenizer(stopwordList, companyList)

    res_df['keyWord'] = res_df['news'].apply(lambda x: extractKeyword(x, tk))
    print(f'{year} keyword extracted')

    output_path = os.path.join(args.home,'data/result',f"{args.year}_result.json")
    res_df.to_json(output_path, orient='records', force_ascii=False, indent=4)
    print(f'{year} Dataframe Saved')
