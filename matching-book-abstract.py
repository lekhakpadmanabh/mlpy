import nltk.stem  
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy as sp

def grab_input():
    titles = []
    descrs = []
    N = int(raw_input())
    for i in xrange(N):
        titles.append(raw_input())
    breaker = raw_input()
    for i in xrange(N):
        descrs.append(raw_input())
    return titles,descrs, N

titles,desc, N  = grab_input()

english_stemmer = nltk.stem.SnowballStemmer('english')
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
    analyzer = super(TfidfVectorizer, self).build_analyzer()
    return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))
tfidf = StemmedTfidfVectorizer(min_df=1, stop_words='english', analyzer='word', ngram_range=(1,3))

dvec = tfidf.fit_transform(desc)

def test(sample):
    svec = tfidf.transform([sample])
    sim = cosine_similarity(svec,dvec)
    return np.argmax(sim)

t_indices = np.zeros(len(titles))
for i,d in enumerate(desc):
    t_index = int(test(titles[i]))
    t_indices[t_index]=i

t_indices = list(map(lambda x: x+1,map(int,t_indices.tolist())))
print '\n'.join(str(p) for p in t_indices) 
