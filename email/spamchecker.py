# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.lda import LDA
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.cluster import KMeans
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearnlab.util import  timer

def get_fullpath(filename):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), filename))

def get_xy():
    rawdata = pd.read_csv(get_fullpath('spambase.data'),
                                          delimiter=',', header=None)
    return np.array(rawdata.icol(range(57))), np.array(rawdata.icol(57))

def select_features(X, y, clf=None, n_features=10):
    if not clf:
        clf = LogisticRegression()
    clf.fit(X, y)
    selector = RFE(clf, n_features_to_select=n_features)
    selector = selector.fit(X, y)
    features = np.array(range(57))
    # print selector.ranking_
    # print selector.support_
    return features[selector.support_]

def make_new_x(rawX, features):
    return  rawX[:, features]

@timer('')
def analysis_selection_train():
    X, y = get_xy()
    # clf = LogisticRegression()
    # clf = LinearRegression()
    clf = LinearSVC()
    print('Origianl features - %s' % clf.__class__.__name__)
    X_train, X_test, Y_train, Y_test, = \
                        train_test_split(X, y,
                                         test_size=0.3, random_state=42)
    clf.fit(X_train, Y_train)
    print clf.score(X_test, Y_test)
    for n in [10, 20, 30, 50, 53, 55]:
        features = select_features(X, y, clf, n)
        new_x = make_new_x(X, features)
        X_train, X_test, Y_train, Y_test, = \
                        train_test_split(new_x, y,
                                         test_size=0.3, random_state=42)
        # print features
        print('=============== New features - (%d) ============' % n)
        clf.fit(X_train, Y_train)
        print clf.score(X_test, Y_test)

def lda(X, y, components=10):
    lda = LDA(n_components=components)
    return lda.fit_transform(X, y)

@timer('')
def analysis_lda_train():
    X, y = get_xy()
    # clf = LogisticRegression()
    # clf = LinearRegression()
    clf = LinearSVC()
    print('Origianl features - %s' % clf.__class__.__name__)
    X_train, X_test, Y_train, Y_test, = \
                        train_test_split(X, y,
                                         test_size=0.3, random_state=42)
    clf.fit(X_train, Y_train)
    print clf.score(X_test, Y_test)
    for n in [10, 20, 30, 50, 53, 55]:
        new_x = lda(X, y, n)
        X_train, X_test, Y_train, Y_test, = \
                        train_test_split(new_x, y,
                                         test_size=0.3, random_state=42)
        # print features
        print('=============== New features - (%d) ============' % n)
        clf.fit(X_train, Y_train)
        print clf.score(X_test, Y_test)

def pca(X, components=10):
    pca = PCA(n_components=components)
    return pca.fit_transform(X)

@timer('')
def analysis_pca_train():
    X, y = get_xy()
    # clf = LogisticRegression()
    # clf = LinearRegression()
    clf = LinearSVC()
    print('Origianl features - %s' % clf.__class__.__name__)
    X_train, X_test, Y_train, Y_test, = \
                        train_test_split(X, y,
                                         test_size=0.3, random_state=42)
    clf.fit(X_train, Y_train)
    print clf.score(X_test, Y_test)
    for n in [10, 20, 30, 50, 53, 55]:
        new_x = pca(X, n)
        X_train, X_test, Y_train, Y_test, = \
                        train_test_split(new_x, y,
                                         test_size=0.3, random_state=42)
        # print features
        print('=============== New features - (%d) ============' % n)
        clf.fit(X_train, Y_train)
        print clf.score(X_test, Y_test)

def scaler(X):
    s = StandardScaler()
    return s.fit_transform(X)

def normalizer(X):
    s = Normalizer(norm='l1')
    return s.fit_transform(X)

@timer('')
def analysis_scaler():
    X, y = get_xy()
    clf = LogisticRegression()
    # clf = LinearRegression()
    # clf = LinearSVC()
    print('Origianl features - %s' % clf.__class__.__name__)
    X_train, X_test, Y_train, Y_test, = \
                        train_test_split(X, y,
                                         test_size=0.3, random_state=42)
    clf.fit(X_train, Y_train)
    print clf.score(X_test, Y_test)
    # new_x = scaler(X)
    new_x = normalizer(X)
    X_train, X_test, Y_train, Y_test, = \
                        train_test_split(new_x, y,
                                         test_size=0.3, random_state=42)
    print('=============== New features - scaler ============')
    clf.fit(X_train, Y_train)
    print clf.score(X_test, Y_test)

def best():
    X, y = get_xy()
    X_train, X_test, Y_train, Y_test, = \
                        train_test_split(X, y,
                                         test_size=0.3, random_state=42)
    lr = LogisticRegression()
    print('======== %s ========' % lr.__class__.__name__)
    lr.fit(X_train, Y_train)
    print lr.score(X_test, Y_test)
    lr_predict = lr.predict(X_test)
    correct_lr_flag = lr_predict == Y_test
    print X_test.shape
    print correct_lr_flag.shape
    correct_lr = X_test[correct_lr_flag]
    print correct_lr.shape

    lsvc = LinearSVC()
    print('======== %s ========' % lsvc.__class__.__name__)
    lsvc.fit(X_train, Y_train)
    print lsvc.score(X_test, Y_test)
    lsvc_predict = lsvc.predict(X_test)
    correct_lsvc_flag = lsvc_predict == Y_test
    print X_test.shape
    print correct_lsvc_flag.shape
    correct_lsvc = X_test[correct_lsvc_flag]
    print correct_lsvc.shape

    print('======== %s ========' % 'enemble cross_section')
    cross_section = np.logical_xor(correct_lr_flag, correct_lsvc_flag)
    only_lr_true_predicted_flag = np.logical_and(cross_section, correct_lr_flag)
    only_lsvc_true_predicted_flag = np.logical_and(cross_section, correct_lsvc_flag)
    only_lr_true_predict = X_test[only_lr_true_predicted_flag]
    only_lsvc_true_predict = X_test[only_lsvc_true_predicted_flag]
    print only_lr_true_predict.shape
    print only_lsvc_true_predict.shape

    lr_y = np.ones(only_lr_true_predict.shape[0])
    lsvc_y = np.zeros(only_lsvc_true_predict.shape[0])
    only_each_predict_X = np.vstack((only_lr_true_predict, only_lsvc_true_predict))
    only_each_predict_y = np.concatenate((lr_y, lsvc_y))

    X_train, X_test, Y_train, Y_test, = \
                        train_test_split(only_each_predict_X, only_each_predict_y,
                                         test_size=0.3, random_state=42)
    advanced_lr = LogisticRegression()
    print('======== enemble %s ========' % advanced_lr.__class__.__name__)
    advanced_lr.fit(X_train, Y_train)
    print advanced_lr.score(X_test, Y_test)

    print('============================')
    print('======== advance ===========')
    X_train, X_test, Y_train, Y_test, = \
                        train_test_split(X, y,
                                         test_size=0.3, random_state=12)

    print('======== %s ========' % lr.__class__.__name__)
    lr.fit(X_train, Y_train)
    print lr.score(X_test, Y_test)
    lr_predict = lr.predict(X_test)
    # correct_lr_flag = lr_predict == Y_test
    # print X_test.shape
    # print correct_lr_flag.shape
    # correct_lr = X_test[correct_lr_flag]
    # print correct_lr.shape

    lsvc = LinearSVC()
    print('======== %s ========' % lsvc.__class__.__name__)
    lsvc.fit(X_train, Y_train)
    print lsvc.score(X_test, Y_test)
    lsvc_predict = lsvc.predict(X_test)
    # correct_lsvc_flag = lsvc_predict == Y_test
    # print X_test.shape
    # print correct_lsvc_flag.shape
    # correct_lsvc = X_test[correct_lsvc_flag]
    # print correct_lsvc.shape

    different_predict_flag = lr_predict != lsvc_predict
    common_predict_flag = lr_predict == lsvc_predict
    print sum(common_predict_flag)
    ss = lr_predict[common_predict_flag] == Y_test[common_predict_flag]
    print sum(ss)
    different_prediction = X_test[different_predict_flag]
    print different_prediction.shape
    new_predict = advanced_lr.predict(different_prediction)
    l = lr_predict[different_predict_flag]
    s = lsvc_predict[different_predict_flag]
    print new_predict.shape
    print l.shape
    print s.shape
    nw = np.array([s[i] if new_predict[i] == 0 else l[i] for i in range(new_predict.shape[0])])
    ll = nw == Y_test[different_predict_flag]
    print sum(ll)
    print sum(ll) + sum(ss)

if __name__ == '__main__':
    # analysis_selection_train()
    # analysis_lda_train()
    # analysis_pca_train()
    best()
