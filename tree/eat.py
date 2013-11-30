# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score

def get_fullpath(filename):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), filename))

def get_xy():
    rawdata = pd.read_csv(get_fullpath('mushrooms.csv'), delimiter = ',')
    length = rawdata.shape[1] 
    le = LabelEncoder()
    labels = []
    s = []
    for i in range(length):
        s.append(le.fit_transform(rawdata.icol(i)))
        labels.append(le.classes_)
    data = np.column_stack(s)
    #print data[:, 0]
    #print data[:, 1:]
    #print labels
    return data[:, 1:], data[:, 0], labels, rawdata.columns

def create_tree():
    return DecisionTreeClassifier(random_state=0)
    
def create_learners():
    return [BernoulliNB(), MultinomialNB(), SVC(), 
                LogisticRegression(), DecisionTreeClassifier(random_state=0)]

def analysis_cv():
    X, y, _, _ = get_xy()
    learners = create_learners()
    size = 2
    for learner in learners:
        print '%s - %f' % (learner.__class__.__name__,
                           sum(cross_val_score(learner, X, y, cv=size))/size)
    
def analysis_with_tree():
    X, y, labels, columns = get_xy()
    tree = create_tree()
    tree.fit(X, y)
    print cross_val_score(tree, X, y, cv=4)
    print tree.n_classes_
    print columns[np.argmax(tree.feature_importances_)]

if __name__ == '__main__':
    #analysis_with_tree()
    analysis_cv()