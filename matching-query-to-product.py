import nltk.stem  
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import LinearSVC

f = open("training.txt")
N = int(f.readline())
results = []
queries = []
for i in xrange(N):
    result,query =  f.readline().split('\t')
    results.append(result)
    queries.append(query[:-1])
results = np.array(results)
queries = np.array(queries)

english_stemmer = nltk.stem.SnowballStemmer('english')
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))
tfidf = StemmedTfidfVectorizer(min_df=1, stop_words='english', analyzer='word', ngram_range=(1,1))

rvec = tfidf.fit_transform(results)
svm = LinearSVC()
svm = svm.fit(rvec.toarray(),queries)

def test(sample):
    svec = tfidf.transform([sample])
    return np.argmax(cosine_similarity(svec,rvec))

N_ = int(raw_input())
for i in xrange(N_):
    svec = tfidf.transform([raw_input()])
    print svm.predict(svec.toarray())[0]

