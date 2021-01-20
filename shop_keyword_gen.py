import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import re
from nltk import ngrams
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from ckiptagger import WS
keyword_num = 10
ws = WS('./data')
def remove(string):
    string = re.sub(r'([0-9])', r'', string)
    string = re.sub(r'[a-zA-z+]', r'', string)
    string = re.sub(r'[^\w\s]', r'', string)
    return string
def keyword_generate(doc):
    doc = ws(doc)
    words = []
    for pair in doc:
        words.append(' '.join(pair))
    cv = CountVectorizer(tokenizer = lambda x: x.split(), max_features = keyword_num)
    try:
        word_count_vector = cv.fit_transform(words)
    except ValueError:                            #no product in this category e.g. media
        return []
    return [word.replace(' ', '') for word in cv.vocabulary_.keys()]
def form_keyword(df, s_keywords = {}):
    info = defaultdict(list)
    for idx, row in enumerate(df.values):
        title = df['title'][idx]
        body_html = df['body_html'][idx]
        slogan = df['slogan'][idx]
        op1 = df['option1'][idx]
        op2 = df['option2'][idx]
        op3 = df['option3'][idx]
        if slogan != slogan:
            slogan = 'None'
        if op1 != op1:
            op1 = 'None'
        if op2 != op2:
            op2 = 'None'
        if op3 != op3:
            op3 = 'None'
        string = str(title) + str(body_html) + str(slogan) + str(op1) + str(op2) + str(op3)
        if s_keywords:
            info = shop_keywords[int(df['shop_Id'][idx])]
            string += info
            string = remove(string)
            info[int(df['id'][idx])].append(string)
        else:
            string = remove(string)
            info[int(df['shop_Id'][idx])].append(string)
    return info
if __name__ == '__main__':
    df = pd.read_csv('./ckip_data.csv')
    shop_info = form_keyword(df)
    shop_keywords = {}
    for idx, shop in enumerate(shop_info.keys()):
        shop_keywords[shop] = keyword_generate(shop_info[shop])
        print('Finished {}nd shop.'.format(idx))
    with open('./shop_keywords.pickle', 'wb') as out:
        pickle.dump(shop_keywords, out)