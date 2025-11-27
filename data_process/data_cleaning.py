import kss
import os
import pandas as pd
import sys
from utils.parser import parse_args, dir_init

# 중복 문장 제거 및 대한항공이 들어간 문장 추출 
def cleanupText(text):

    # 문장 분리 
    sentences = tokenizer = kss.split_sentences(text)
    duplicates = []
    cleaned = []
    for s in sentences:
        if s in cleaned:
            if s in duplicates:
                continue
            else:
                duplicates.append(s)
        else:
            cleaned.append(s)
    text = ' '.join(cleaned)
    result = [x for x in sentences if '대한항공' in x]
    return (text, result)

def remove_news(args):
    file_path = os.path.join(args.home, "data_process", "news_data", f"news_{args.year}.json")
    df = pd.read_json(file_path)

    # 대한항공 주 스포츠 제외 
    df = df[df["content"].str.contains("배구")==False]
    df = df[df["title"].str.contains("배구")==False]
    df = df[df["content"].str.contains("탁구")==False]

    # 스포츠 언론사 제외 
    df = df[df["publisher"].str.contains("스포츠")==False]

    # 부고소식 제외 
    df = df[df["title"].str.contains("[부고]")==False]
    df = df[df["title"].str.contains("별세")==False]
    df = df[df["content"].str.contains("별세")==False]
    df = df[df["content"].str.contains("장례식")==False]

    # 특정 언론사 제외
    df = df[df["publisher"].str.contains("뉴시스")==False]
    df = df[df["publisher"].str.contains("OSEN")==False]
    df = df[df["publisher"].str.contains("스포티비뉴스")==False]
    df = df[df["publisher"].str.contains("위키트리")==False]

    df["content"].apply(lambda x: cleanupText(x))

    output_path = os.path.join(args.home, "data_process", "data", f"news_{args.year}_update.json")
    df.to_json(output_path, orient='records', force_ascii=False, indent=4)