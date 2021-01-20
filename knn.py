import pandas as pd
import pickle
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

n = 3
MAX_VOCAB = 999

def form_docs(df):
    docs = []
    #make docs
    for row in df.values:
        doc = []
        for idx, inst in enumerate(row[2:6]):
            if inst != inst:
                inst = '<NAN>'
            elif idx == 1 or idx == 2:                                       #title, body_html
                string = inst.replace('\n', '')
                string = re.sub(r'[a-zA-z+]', r'', string)
                string = re.sub(r'[^\w\s]', r'', string)
                string = re.sub(r'[0-9]+', r'', string)
                string = string.replace(' ', '')
                string = re.sub(r'(.)', '\g<1> ', string)
                doc.append(string)
            else:
                doc.append(inst)
            sent = ' '.join(doc)
        docs.append(sent)
        del sent, doc, string 
    return docs
if __name__ == '__main__':
    
    print('Opening Data...')
    with open('./labelled.pickle', 'rb') as file:
        df = pickle.load(file)
    
    print('Forming Tfidf Table...')
    #form tfidf table   
    corpus = form_docs(df) 
    cv = CountVectorizer(max_features = MAX_VOCAB, tokenizer = lambda x: x.split(), ngram_range = (1, n))
    word_count_vector = cv.fit_transform(corpus)
    tfidf_trans = TfidfTransformer(smooth_idf = True,use_idf = True) 
    tfidf_trans.fit(word_count_vector)
    del word_count_vector, corpus
    
    print('Cleansing Data...')
    training = df.drop(df[df['labels'] == -1].index)
    training = training.drop(training[training['price'] == 0].index)
    training = training.drop(training[training['price'] > 1000000].index)
    
    print('Forming Training Tfidf Vectors...')
    t_docs = form_docs(training)
    wordcount = cv.transform(t_docs)
    X_tfidf = tfidf_trans.transform(wordcount)
    del wordcount, t_docs

    print('Normalizing Training Price...')
    norm_price = []
    for idx, inst in enumerate(training['price']):
        if inst > 1.7976931348623157e+308:
            norm_price.append(1)
        else:
            norm_price.append(1 /(1 + np.exp(-inst)))
    X = np.append(X_tfidf.toarray(), np.array(norm_price)[:, np.newaxis], axis = 1)
    y = training['labels']
    
    print('Start Training Process...')
    neigh = KNeighborsClassifier(n_neighbors = 5)
    neigh.fit(X, y)
    del X_tfidf, norm_price
    print('Finished Training')
    
    print('Forming Testing Tfidf Vectors...')
    testing = df.drop(df[df['labels'] != -1].index)
    test_docs = form_docs(testing)
    t_wordcount = cv.transform(test_docs)
    test_tfidf = tfidf_trans.transform(t_wordcount)
    del t_wordcount, test_docs
    
    print('Normalizing Testing Price...')
    t_norm_price = []
    for idx, inst in enumerate(testing['price']):
        if inst > 1.7976931348623157e+308:
            t_norm_price.append(1)
        else:
            t_norm_price.append(1 /(1 + np.exp(-inst)))
    X = np.append(test_tfidf.toarray(), np.array(t_norm_price)[:, np.newaxis], axis = 1)
    del t_norm_price

    print('Start Label Prediction...')
    pred = neigh.predict(X)
    result = {'product_id':testing['id'], 'category_id':pred}
    result = pd.DataFrame(result)
    
    print('Saving Prediction...')
    with open('./result.pickle', 'wb') as out:
        pickle.dump(result, out)
    print('Process Finished.')
