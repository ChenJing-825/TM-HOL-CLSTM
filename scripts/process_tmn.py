encoding = 'UTF-8'
import numpy as np
import gensim
from scipy import sparse
import pickle
import json
import pandas as pd
import csv
import nltk
import logging
import copy
import os
from nltk.stem.porter import *
from feature import *
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO

data_file = '../data/tm-clstm/labeled_data.csv'
data_dir = os.path.dirname(data_file)
df = pd.read_csv('../data/tm-clstm/labeled_data.csv', encoding='utf8')
tweets = df['tweet'].values
tweets = [x for x in tweets if type(x) == str]
tweets_class = df['class'].values
tweets_class_list = tweets_class.tolist()
len_label = len(tweets_class_list)
label_dict = {}
for i in range(len_label):
    if tweets_class_list[i] == 0:
        label_dict['0'] = 'Hate speech'
    elif tweets_class_list[i] == 1:
        label_dict['1'] = 'offensive_language'
    else:
        label_dict['2'] = 'neither'

stopwords = stopwords = nltk.corpus.stopwords.words("english")
other_exclusions = ["#ff", "ff", "rt", "ipad", "android", "iphone", "xxx"]
stopwords.extend(other_exclusions)
space_pattern = r'\s+'
giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                       '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
mention_regex = r'@[\w\-]+'

length = len(tweets)

for a in range(0, length):
    parsed_text = tweets[a]
    parsed_text = re.sub(space_pattern, ' ', parsed_text)
    parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
    parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
    parsed_text = re.sub(mention_regex, ' ', parsed_text)
    tweet = " ".join(re.split("[^a-zA-Z]*", parsed_text.lower())).strip()
    stemmer = PorterStemmer()
    tweets_token = [stemmer.stem(t) for t in tweet.split()]
    tweets[a] = tweets_token

dictionary = gensim.corpora.Dictionary(tweets)
bow_dictionary = copy.deepcopy(dictionary)
bow_dictionary.filter_tokens(list(map(bow_dictionary.token2id.get, stopwords)))
len_1_words = list(filter(lambda w: len(w) == 1, bow_dictionary.values()))
bow_dictionary.filter_tokens(list(map(bow_dictionary.token2id.get, len_1_words)))
bow_dictionary.filter_extremes(no_below=3, keep_n=None)
bow_dictionary.compactify()


def get_wids(text_doc, seq_dictionary, bow_dictionary, ori_labels):
    seq_doc = []
    tweets_feat_text=[]
    row = []
    col = []
    value = []
    row_id = 0
    m_labels = []
    for d_i, doc in enumerate(text_doc):
        if len(bow_dictionary.doc2bow(doc)) < 3:
            continue
        else:
            tweets_feat_text.append(doc)
        for i, j in bow_dictionary.doc2bow(doc):
            row.append(row_id)
            col.append(i)
            value.append(j)
        row_id += 1
        wids = list(map(seq_dictionary.token2id.get, doc))
        wids = np.array(list(filter(lambda x: x is not None, wids))) + 1
        m_labels.append(ori_labels[d_i])
        seq_doc.append(wids)
    lens = list(map(len, seq_doc))
    bow_doc = sparse.coo_matrix((value, (row, col)), shape=(row_id, len(bow_dictionary)))
    logging.info("get %d docs, avg len: %d, max len: %d" % (len(seq_doc), np.mean(lens), np.max(lens)))
    return seq_doc, bow_doc, m_labels, tweets_feat_text


seq_title, bow_title, label_title, tweets_feat_text = get_wids(tweets, dictionary, bow_dictionary, tweets_class)

feat_text = []
for i in range(len(tweets_feat_text)):
    list2 = [str(j) for j in tweets_feat_text[i]]
    list3 = ' '.join(list2)
    feat_text.append(list3)
for i in range(len(tweets_feat_text)-1):
    maxlen = 0
    if len(feat_text[i]) > maxlen:
        maxlen = len(feat_text[i])

feat_array = get_oth_features(feat_text)

indices = np.arange(len(seq_title))
np.random.shuffle(indices)

nb_test_samples = int(0.2 * len(seq_title))
seq_title = np.array(seq_title)[indices]
seq_title_train = seq_title[:-nb_test_samples]
seq_title_test = seq_title[-nb_test_samples:]

bow_title = bow_title.tocsr()
bow_title = bow_title[indices]
bow_title_train = bow_title[:-nb_test_samples]
bow_title_test = bow_title[-nb_test_samples:]


label_title = np.array(label_title)[indices]
label_title_train = label_title[:-nb_test_samples]
label_title_test = label_title[-nb_test_samples:]

feat_title = feat_array
feat_title_train = feat_title[:-nb_test_samples]
feat_title_test = feat_title[-nb_test_samples:]

# save
logging.info("save data...")
pickle.dump(seq_title, open(os.path.join(data_dir, "dataMsg"), "wb"))
pickle.dump(seq_title_train, open(os.path.join(data_dir, "dataMsgTrain"), "wb"))
pickle.dump(seq_title_test, open(os.path.join(data_dir, "dataMsgTest"), "wb"))

pickle.dump(bow_title, open(os.path.join(data_dir, "dataMsgBow"), "wb"))
pickle.dump(bow_title_train, open(os.path.join(data_dir, "dataMsgBowTrain"), "wb"))
pickle.dump(bow_title_test, open(os.path.join(data_dir, "dataMsgBowTest"), "wb"))

pickle.dump(label_title, open(os.path.join(data_dir, "dataMsgLabel"), "wb"))
pickle.dump(label_title_train, open(os.path.join(data_dir, "dataMsgLabelTrain"), "wb"))
pickle.dump(label_title_test, open(os.path.join(data_dir, "dataMsgLabelTest"), "wb"))

pickle.dump(feat_title, open(os.path.join(data_dir, "dataMsgFeatTrain"), "wb"))
pickle.dump(feat_title_train, open(os.path.join(data_dir, "dataMsgFeatTrain"), "wb"))
pickle.dump(feat_title_test, open(os.path.join(data_dir, "dataMsgFeatTest"), "wb"))

dictionary.save(os.path.join(data_dir, "dataDictSeq"))
bow_dictionary.save(os.path.join(data_dir, "dataDictBow"))
json.dump(label_dict, open(os.path.join(data_dir, "labelDict.json"), "w"), indent=4)
logging.info("done!")
