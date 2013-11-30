# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer#, TfidVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_curve, auc, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import ShuffleSplit, cross_val_score


from sklearnlab.util import convert2d, plot_precision_recall,\
                            plot_confusion_matrix, plot_roc,\
                            plot_score, Logger

def get_fullpath(filename):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), filename))

def create_xy():
    rawdata = pd.read_csv(get_fullpath('rawsocialpick.csv'), delimiter = '\t')
    #rawdata = pd.read_csv('nounsocialpick.csv', delimiter = '\t')
    
    data = np.array(rawdata)
    y_size = data.shape[0]
    print y_size
    split_size = y_size / 10
    print split_size
    le = LabelEncoder()
    y = le.fit_transform(data[:, 0])
    #print(le.classes_)

    #tfidf = TfidVectorizer(ngram_range=(1, 3))
    vectorizer = CountVectorizer(min_df=1)
    X = vectorizer.fit_transform(data[:, 2])

    X_train, Y_train = X[:split_size], y[:split_size]
    X_test, Y_test = X[split_size:], y[split_size:]
    return X_train, Y_train, X_test, Y_test, le.classes_

def get_xy():
    #rawdata = pd.read_csv('rawsocialpick.csv', delimiter = '\t')
    rawdata = pd.read_csv(get_fullpath('nounsocialpick.csv'), delimiter = '\t')
    
    data = np.array(rawdata)
    le = LabelEncoder()
    y = le.fit_transform(data[:, 0])
    vectorizer = CountVectorizer(min_df=1)
    X = vectorizer.fit_transform(data[:, 2])
    return X, y
    
def create_learners():
    return [BernoulliNB(), MultinomialNB(), SVC(), LogisticRegression()]

def create_learner():
    return BernoulliNB()

def analysis_cv():
    X, y = get_xy()
    learners = create_learners()
    size = 2
    for learner in learners:
        print '%s - %f' % (learner.__class__.__name__,
                           sum(cross_val_score(learner, X, y, cv=size))/size)
    
def analysis_with_learners():
    X_train, Y_train, X_test, Y_test, targetname = create_xy()
    learners = create_learners()
    scores = [l.fit(X_train, Y_train).score(X_test, Y_test) for l in learners]
    plot_score(learners, scores)

def analysis_multi():
    X_train, Y_train, X_test, Y_test, targetname = create_xy()
    learners = create_learner()
    learners.fit(X_train, Y_train)
    print('============== multi ==============')
    print('Score %f' % learners.score(X_test, Y_test))
    cm = confusion_matrix(Y_test, learners.predict(X_test))
    print('Confusion Matrix')
    print(cm)
    plot_confusion_matrix(cm, targetname)
    
def analysis_1vsN():
    print('============== One Vs Rest ==============')
    X_train, Y_train, X_test, Y_test, targetname = create_xy()
    clf = create_learner()
    clf.fit(X_train, Y_train)
    cm = confusion_matrix(Y_test, clf.predict(X_test))
    for i in range(3):
        print convert2d(cm, i)
        yy = np.asarray(Y_test==i, dtype=int)
        prob = clf.predict_proba(X_test)[:,i]
        ## precision-recall        
        precision, recall, threld = precision_recall_curve(yy, prob)
        area = auc(recall, precision)
        plot_precision_recall(precision, recall, area, targetname[i])
        ## roc
        fpr, tpr, roc_thresholds = roc_curve(yy, prob)
        area = auc(fpr, tpr)
        plot_roc(area, tpr, fpr, targetname[i])

"""
cv = ShuffleSplit(n=len(X), n_iter=10, test_size=0.3, indices=True, random_state=0)
for train, test in cv:
    X_train, y_train = X[train], y[train]
    X_test, y_test = X[test], y[test]
    print X_train
    print X_test
"""
if __name__ == '__main__':
    l = Logger(targetstdout=True, targetfile=True)
    l.write('00')    
    analysis_cv()
    l.write('11')
    #analysis_with_learners()
    #analysis_multi()
    #analysis_1vsN()