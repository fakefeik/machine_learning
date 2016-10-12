from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

newsgroups = datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])

vectorizer = TfidfVectorizer()
ng = vectorizer.fit_transform(newsgroups['data'])

def most_informative_feature_for_class_svm(vectorizer, classifier,  n=10):
    labelid = 0
    feature_names = vectorizer.get_feature_names()
    svm_coef = classifier.coef_.toarray()
    topn = sorted(zip(abs(svm_coef[labelid]), feature_names))[-n:]

    for coef, feat in topn:
        yield feat
        print(feat, coef)

clf = SVC(C=1.0, kernel='linear', random_state=241)
clf.fit(ng, newsgroups['target'])
fn = vectorizer.get_feature_names()
f = list(most_informative_feature_for_class_svm(vectorizer, clf))
print(' '.join(sorted(f)))
