# -*- coding: utf-8 -*-
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import persist_cv_file
from sklearn import datasets
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

english_stemmer = nltk.stem.SnowballStemmer('english')
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))


class NoNameBaseEstimatro(BaseEstimator):
    def __init__(self):
        pass
    
    def get_feature_name(self):
        pass
    
    def transform(self):
        pass
    
    def fit(self, data):
        return self

class BaseMajorityEstimator(BaseEstimator, ClassifierMixin):
    def _fit(self, X, y):
        from scipy.stats import randint
        randidx = randint.rvs(0, len(y), size=10)
        counts = np.bincount(randidx)
        self.majority_ = np.argmax(counts)
        
class MajorityEstimator(BaseMajorityEstimator):
    def __init__(self, weight=3):
        self.weight = weight
    
    def fit(self, X, y):
        self._fit(X, y)
        return self
        
    def predict(self, X):
        return np.repeat(self.majority_, len(X)) * self.weight

def test_grid_search(clf, X, y):
    parameters = {'MajorityEstimator__weight':[1, 2]}    
    gs = GridSearchCV(clf, parameters)
    gs.fit(X, y)
    print gs.best_estimator_.get_params()
    
if __name__ == '__main__':
    """
    vectorizer = StemmedCountVectorizer(min_df=1, stop_words='english')
    l = ['I go to home', 'I went to the school', 'Let me go']
    print vectorizer.fit_transform(l)
    print vectorizer.get_feature_names()
    vectorizer = StemmedTfidfVectorizer(min_df=1, stop_words='english')
    print vectorizer.fit_transform(l)
    print vectorizer.get_feature_names()
    
    # test for customized estimator
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    majorestimator = MajorityEstimator()
    majorestimator.fit(X, y)
    print  majorestimator.predict(X[1:4,:])
    
    # test for pipeline
    pca = PCA()
    majorestimator = MajorityEstimator()
    clf = Pipeline(steps=[('pca', pca), ('MajorityEstimator', majorestimator)])
    clf.fit(X, y)
    print clf.predict(X)
    
    # test for grid_search
    noestimator = MajorityEstimator()
    parameters = {'weight':[1, 2]}    
    gs = GridSearchCV(clf, parameters)
    gs.fit(X, y)
    print gs.best_estimator_.get_params()
    
    # test for pipe with grid_search
    pca = PCA()
    majorestimator = MajorityEstimator()
    clf = Pipeline(steps=[('pca', pca), ('MajorityEstimator', majorestimator)])
    parameters = {'MajorityEstimator__weight':[1, 2]}        
    gs = GridSearchCV(clf, parameters)
    gs.fit(X, y)
    print gs.best_estimator_.get_params()
    """
    
    from sklearn.datasets import load_iris
    iris = load_iris()
    X, y = iris.data, iris.target
    cv = KFold(X.shape[0], 2, shuffle=True, random_state=0)
    persist_cv_file(X, y, cv)   
