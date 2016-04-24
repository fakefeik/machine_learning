from json import load, dump
from random import seed, shuffle

from nltk import download
from nltk.corpus import stopwords
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

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
stop_words = stopwords.words('russian')

true_k = len(target_names)

vectorizer = TfidfVectorizer(max_df=0.5, max_features=n_features,
                             min_df=2, stop_words=stop_words)
X = vectorizer.fit_transform(data)

svd = TruncatedSVD(n_components, random_state=random_state)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)

X = lsa.fit_transform(X)

km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                     init_size=1000, batch_size=1000, random_state=random_state)
# km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
#             random_state=random_state)

km.fit(X)

original_space_centroids = svd.inverse_transform(km.cluster_centers_)
order_centroids = original_space_centroids.argsort()[:, ::-1]

terms = vectorizer.get_feature_names()

latent = 'Top terms per cluster:'
for i in range(true_k):
    latent += '\nCluster {}'.format(i)
    for ind in order_centroids[i, :20]:
        latent += ' {}'.format(terms[ind])

pred = km.predict(X)
l = [""]*len(data)
for i in range(len(data)):
    l[inverse_index[i]] = pred[i]

with open('boards/latent.txt', 'w+') as f:
    f.write(latent)

with open('boards/data.txt', 'w+') as f:
    f.write('\n'.join(map(lambda x: "{}: {}".format(x, l[x]), range(len(l)))))
