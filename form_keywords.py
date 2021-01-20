import numpy as np
import pandas as pd
import pickle
import re
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

class form_keywords:
    def __init__(self, location, keyword_num = 10, n = tuple((2, 4))):
        '''
        location : The file you want to form keywords (csv file)
        '''
        self.n = n
        self.keyword_num = keyword_num
        c_df = pd.read_excel('./data/google_categories.xls')
        self.categories = set(c_df['Animals & Pet Supplies'])
        self.cid_to_name = dict([(row[0], row[1]) for row in c_df.values])
        df = pd.read_csv(location)
        df = df.drop(df[df['category_id'] == '未分類'].index)
        self.df = df.drop(df[df['title'] != df['title']].index)                  #remove nan on title
    def remove(self, string):
        string = re.sub(r'(.)', r'\g<1> ', string)
        string = re.sub(r'([0-9])', r'', string)
        string = re.sub(r'[a-zA-z+]', r'', string)
        string = re.sub(r'[^\w\s]', r'', string)
        return string
    def keyword_generate(self, doc):
        doc = [self.remove(string) for string in doc] 
        cv = CountVectorizer(max_features = self.keyword_num, 
             tokenizer = lambda x: x.split(), ngram_range = (self.n[0], self.n[1]))
        try:
            word_count_vector = cv.fit_transform(doc)
        except ValueError:                            #no product in this category e.g. media
            return []
        return [word.replace(' ', '') for word in cv.vocabulary_.keys()]
    def fit(self):
        docs = defaultdict(list)
        key_word_table = defaultdict(list)
        for row in self.df.values:
            idx = int(row[10])
            docs[self.cid_to_name[idx]].append(row[9])                             #form doc according to different categories
        for cat in self.categories:
            key_word_table[cat] = self.keyword_generate(docs[cat])
        return key_word_table