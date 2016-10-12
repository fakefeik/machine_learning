from json import load, dump
from random import seed, shuffle

import numpy as np
import lda
import lda.datasets
from nltk import download
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from stop_words import get_stop_words

from collect import collect, remove_common_words

n_components = 1000
n_features = 10000

random_state = 1


target_names = ['pr', 'po', 'au']

try:
    with open('boards/data.json', 'r', encoding='utf8') as f:
        data = load(f)
except FileNotFoundError:
    d = collect(target_names, 100, random_state)
    data = []
    for k in d:
        for e in d[k]:
            data.append(remove_common_words(e))
    with open('boards/data.json', 'w+', encoding='utf8') as f:
        dump(data, f, ensure_ascii=False)

data2 = data.copy()

seed(random_state)
shuffle(data)
inverse_index = []
for i in range(len(data)):
    inverse_index.append(data2.index(data[i]))

download("stopwords")
stop_words = stopwords.words('russian') + get_stop_words('russian') + [
    "которые", "тебе", "вообще", "который", "чему", "очень", "могут", "например", "столько", "делать", "думаю", "будут",
    "сделать", "знаю", "хочу", "вроде", "спасибо", "прос", "лько", "вроде", "нных", "нные", "могу", "нахуй", "нихуя",
    "корые", "корый", "какие", "кстати", "типа", "сразу"
]
true_k = len(target_names)

vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=n_features,
                             stop_words=stop_words)
X = vectorizer.fit_transform(data)

km = lda.LDA(n_topics=3, n_iter=2000, random_state=random_state)
km.fit(X)

n_top_words = 20

vocab = vectorizer.get_feature_names()
latent = ""
for i, topic_dist in enumerate(km.topic_word_):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    latent += 'Topic {}: {}\n'.format(i, ' '.join(topic_words))

with open('boards/latent.txt', 'w+', encoding='utf8') as f:
    f.write(latent)

topics = [""]*len(data)
doc_topic = km.doc_topic_
for i in range(0, len(data)):
    topics[inverse_index[i]] = doc_topic[i]


def get_s(arr):
    s = ""
    for i, e in enumerate(arr):
        if e >= 0.05:
            s += '\t{}: {}\n'.format(i, e)
    return s

with open('boards/data.txt', 'w+') as f:
    f.write('\n'.join(map(lambda x: "{}:\n{}".format(x, get_s(topics[x])), range(len(topics)))))
