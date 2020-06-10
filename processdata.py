from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

import os
import re
import string
from math import log
from pathlib import Path
from typing import List

from transformers import pipeline
import nltk
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from lib.config import DataConfig
from vnnlp.vietnamnlp import vn_tokenize

def prep_data(remake: bool) -> DataConfig:
    curr_directory = str(Path.cwd()) 
    data_directory = curr_directory + "/data"
    stock_directory = Path(data_directory + "/stock-data")

    try:
        for f in stock_directory.rglob('*-data.csv'):
            # TODO: Support multiples csv files
            stock_data = f
            print('Stock data used:', f)
    except:
        print('There is no .csv stock data')

    try:
        for n in stock_directory.rglob('*-news.json'):
            news_data = n
            print('News data used:', n)
    except:
        print('There is no .json news')

    try:
        for n in stock_directory.rglob('*-eps.csv'):
            eps_data = n
            print('Eps data used:', n)
    except:
        print('There is no .csv eps data')


    train_path = data_directory + '/train' + '/train_data.npy'
    test_path = data_directory + '/test' + '/test_data.npy'
    val_path = data_directory + '/val' + '/validation_data.npy'
    feature = ['Close', 'sentiment', 'vn_embedding', 'Report_EPS', 'Datetime', 'Label']

    start_date = '2009-12-31'
    end_date = '2020-04-21'
    train_index = 1500
    test_index = 2500

    def _sentiment_transformer(text_df: pd.DataFrame) -> List[int]:
        print('Starting sentiment-analysis...')
        nlp = pipeline('sentiment-analysis')
        length = len(text_df['en_text'])
        dummy = []

        for i in range(length):
            if text_df['en_text'][i] is not None:
                ret = nlp(text_df['en_text'][i])
                if ret[0]['label'] == 'NEGATIVE':
                    dummy.append(-1)
                elif ret[0]['label'] == 'POSITIVE':
                    dummy.append(1)
                else:
                    dummy.append(0)
            else:
                dummy.append(0)
        print('Completing sentiment-analysis...')
        return dummy

    def _create_label(csv_location: str) -> None:
        df = pd.read_csv(csv_location)
        if 'Label' not in df.columns:
            dat = df['Close'].values
            true_label = [0]*len(dat)

            tolerance = 0.0001
            for i in range(len(dat)):
                if i == 0 or i == len(dat)-1:
                    true_label[i] = "hold"
                else:
                    val_t = dat[i]
                    val_t_1 = dat[i+1]
                    upper_bound = val_t + tolerance*val_t
                    lower_bound = val_t - tolerance*val_t

                    if upper_bound < val_t_1:
                        true_label[i] = "buy"
                    elif lower_bound > val_t_1:
                        true_label[i] = "sell"
                    else:
                        true_label[i] = "hold"
            df['Label'] = true_label
            df.to_csv(csv_location, index=None)
        else:
            print('Label exists, skipping create label process')

    def _load_data() -> pd.DataFrame:
        print('Preparing train data, test data and validation data...')

        _create_label(stock_data)
        vnm_df = pd.read_csv(stock_data, delimiter=",", encoding='utf-8')
        vnm_df.columns = ['Ticker', 'Datetime', 'Close', 'Volume', 'Label']
        vnm_df['Datetime'] = pd.to_datetime(vnm_df.Datetime, format='%Y%m%d')

        news_df = pd.read_json(news_data, encoding='utf-8')
        news_df.columns = ['title', 'text', 'Datetime', 'en_text']
        news_df = news_df[news_df['text'].notna()]
        news_df = news_df[news_df['en_text'].notna()] # Careful this line will change the number of instances of embedding features 716 to 300 feautures
        news_df['Datetime'] = pd.to_datetime((pd.to_datetime(news_df['Datetime']).dt.strftime('%Y-%m-%d'))) # remove hour minute second
        news_df = news_df.loc[(news_df['Datetime'] > start_date) & (news_df['Datetime'] <= end_date)] # mask from 31/12/2009 to 2020-04-21

        eps_df = pd.read_csv(eps_data, delimiter=",", encoding='utf-8')
        eps_df.columns = ['BCTC', 'LNSTTNDN', 'CPDLH', 'Report_EPS', 'Datetime']
        eps_df.drop(['BCTC', 'LNSTTNDN', 'CPDLH'], axis=1, inplace=True)
        eps_df['Datetime'] = pd.to_datetime(eps_df.Datetime, format='%Y%m%d')
        eps_df = eps_df.loc[(eps_df['Datetime'] > start_date) & (eps_df['Datetime'] <= end_date)]

        items = vnm_df['Datetime'].to_list()

        def overlapping(items: List, pivot: str):
            minimum = pd.Timedelta(100000, unit='d') # assume 100000 (100 years) is longest, if difference of timeseries is bigger, this will fail no doubt
            for elem in items:
                if pivot - elem >= pd.Timedelta(0):
                    curr = pivot - elem
                    minimum = min(minimum, curr)
                    if minimum == curr:
                        ret = elem
            if minimum == pd.Timedelta(100000, unit='d'):
                ret = None
            return ret

        news_df['Datetime'] = [overlapping(items, pivot) for pivot in news_df['Datetime']]
        news_df = news_df[news_df['Datetime'].notna()]

        eps_df['Datetime'] = [overlapping(items, pivot) for pivot in eps_df['Datetime']]
        eps_df = eps_df[eps_df['Datetime'].notna()]

        add_sentence = lambda a: " ".join(a)
        news_df = news_df.groupby(by='Datetime').agg(
            {'title': add_sentence,
            'text': add_sentence,
            'en_text': add_sentence}
        ).reset_index() # 647 not 716 since eng-news
        news_df['sentiment'] = _sentiment_transformer(news_df)
        news_df, n_component = _text_embedding(news_df)

        vnm_df = vnm_df.merge(eps_df, on='Datetime', how='left')
        vnm_df['Report_EPS'] = vnm_df['Report_EPS'].fillna(value=0)

        df = vnm_df.merge(news_df, on='Datetime', how='left')
        df['sentiment'] = df['sentiment'].fillna(value=0)
        for row in df.loc[df.vn_embedding.isnull(), 'vn_embedding'].index:
            df.at[row, 'vn_embedding'] = [0]*n_component # update this number here
        df['vn_embedding'] = [np.array(elem, dtype=np.float32) for elem in df['vn_embedding']]

        # df['Close'] = pd.to_numeric(df['Close'].apply(lambda x:  x.replace(',','.'))) # convert close price string to float64
        df.drop(['title', 'text', 'en_text'], axis=1, inplace=True) # check this line when embedding

        # code for colab training
        ########
        # buffr = df['vn_embedding'] 
        # buffr.to_csv('embedding.csv')
        # df.to_csv('vn_embedding.csv')
        # np.save('vn_embedding.npy', buffr)
        ########

        return df

    def _text_embedding(df: pd.DataFrame) -> pd.DataFrame:
        df['text'] = [elem.strip() for elem in df['text']] # remove leading and trailing whitespace
        df['text'] = [re.sub(' +', ' ', elem) for elem in df['text']] # remove whitespace 2 or more in middle
        df['text'] = [re.sub('\n|\r', ' ', elem) for elem in df['text']] # remove line break 
        df['text'] = [elem.translate(str.maketrans('', '', string.punctuation)) for elem in df['text']] # remove punctuation

        os.chdir("./vnnlp") # move to /vnnlp folder to make use of model and jar file
        df = vn_tokenize(df)
        os.chdir("..")

        data_root_list = df['tokenized-text'].values
        corpus = {}
        for i in range(len(data_root_list)):
            corpus['corpus_{}'.format(i)] = data_root_list[i]        

        # Calculating word set
        word_set = []
        for key, val in corpus.items():
            word_set += val

        # union of the word_set
        word_set = list(set(word_set))

        # Compute word dictionary with Term frequency algorithms
        def compute_tf(word_dict_no, corpus_data_no):
            tf = {}

            sum_corpus_no = len(corpus_data_no) + 0.000000001  # avoid case 0 divider
            for word, count in word_dict_no.items():
                tf[word] = round(count / sum_corpus_no, 3)

            return tf

        # Compute word dictionary with Inverse Document Frequency algorithms
        def compute_idf(word_dict_for_idf):
            n = len(word_dict_for_idf)
            idf  = dict.fromkeys(word_dict_for_idf['word_dict_0'].keys(), 0)

            for i in range(n):
                for word, count in word_dict_for_idf['word_dict_{}'.format(i)].items(): # get word_dict_no
                    if count > 0:
                        idf[word] += 1

            for word, count in idf.items():
                idf[word] = round(log(n/float(count)), 3)

            return idf

        # Compute term frequency dictionary with Term Frequency-Inverse Document Frequency algorithms
        def compute_tf_idf(tf_dict, idf):

            tf_idf = dict.fromkeys(tf_dict.keys(), 0)
            for word, count in tf_dict.items():
                tf_idf[word] = round(count * idf[word], 3)

            return tf_idf

        word_dict = {}

        for i in range(len(corpus)):
            word_dict['word_dict_{}'.format(i)] = dict.fromkeys(word_set, 0)

        for i in range(len(corpus)):
            for word_corpus_data in corpus['corpus_{}'.format(i)]:
                word_dict['word_dict_{}'.format(i)][word_corpus_data] += 1

        tf_dict = {}

        for i in range(len(corpus)):
            tf_dict['tf_dict_{}'.format(i)] = compute_tf(word_dict['word_dict_{}'.format(i)], corpus['corpus_{}'.format(i)])
        
        idf = compute_idf(word_dict)
        tf_idf_dict = {}

        for i in range(len(corpus)):
            tf_idf_dict['tf_idf_dict_{}'.format(i)] = compute_tf_idf(tf_dict['tf_dict_{}'.format(i)], idf)

        # Create a tf-idf vectorization array
        tfidf_vectorization = []

        for tf_idf_dict_key, tf_idf_dict_val in tf_idf_dict.items():
            tfidf_vectorization.append(list(tf_idf_dict_val.values()))
          
        tfidf_vectorization = np.array(tfidf_vectorization)
        print('Shape of tfidf_vectorization: {}'.format(tfidf_vectorization.shape))

        ##### Scale directly the data here for PCA, since PCA takes too much time, I put it here instead of scaling during the training
        scaler = MinMaxScaler()
        X_train = tfidf_vectorization[:train_index]
        std_scale = scaler.fit(X_train)
        data_rescaled = scaler.transform(tfidf_vectorization)

        pca = PCA(n_components=0.95)
        pca.fit(data_rescaled)
        tfidf_vectorization = pca.transform(data_rescaled)
        print('Shape of pca tfidf_vectorization: {}'.format(tfidf_vectorization.shape))
        n_component = tfidf_vectorization.shape[1]
        #####
        print("Completing news-embedding...")
        df['vn_embedding'] = [elem for elem in tfidf_vectorization] # careful


        return df, n_component

    def _select_feature(df_data: pd.DataFrame, feature_list: List[str]) -> None:
        features = df_data[feature_list].values

        print("Number of vnm instances: ", features.shape[0])
        print("Number of vnm feature: ", features.shape[1]) # 0: price, 1: sentiment

        train_vnm = features[:train_index]
        test_vnm = features[train_index:test_index]
        val_vnm = features[test_index:]

        np.save(train_path, train_vnm)
        np.save(test_path, test_vnm)
        np.save(val_path, val_vnm)

    if remake:
        if os.path.exists(train_path) and os.path.exists(test_path) and os.path.exists(val_path):
            print('Deleted existed file...')
            os.remove(train_path)
            os.remove(test_path)
            os.remove(val_path)

            df = _load_data()
            _select_feature(df, feature)
        else:
            print('There is no existed file, creating new...')
            df = _load_data()
            _select_feature(df, feature)
    else:
        if os.path.exists(train_path) and os.path.exists(test_path) and os.path.exists(val_path):
            print('Skipping the prepare data process...')
        else:
            print('There is no existed file, cannot skip the prepare data process, auto proceed...')
            df = _load_data()
            _select_feature(df, feature)

    data_config = DataConfig(train_path, test_path, val_path)

    return data_config

