# -*- coding: utf-8 -*-
import os
import sys
import logging
import numpy as np
import pandas as pd
from matplotlib import pylab
from sklearn.externals import joblib
from sklearn.grid_search import IterGrid
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix
from sklearnlab.util.base import StemmedCountVectorizer,\
                                 StemmedTfidfVectorizer

__all__ = ['StemmedCountVectorizer', 'StemmedTfidfVectorizer',
           'Logger', 'get_projectpath', 'convert2d', 'plot_score',
           'plot_precision_recall', 'plot_confusion_matrix', 'plot_roc',
           'timer', 'check_na'
            ]

import time
def timer(label='', trace=True):
    def onDecorator(func):
        def onCall(*args, **kwargs):
            start = time.clock()
            result = func(*args, **kwargs)
            elapsed = time.clock() - start
            if trace:
                format = '%s%s: %.5f'
                values = (label, func.__name__, elapsed)
                print(format % values)
            return result
        return onCall
    return onDecorator

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Logger(object):
    __metaclass__ = Singleton
    def __init__(self, targetstdout=True, targetfile=False, filepath='out.log'):
        self.log = True if any([targetfile,targetstdout]) else False
        log_formatter = logging.Formatter('%(asctime)s %(message)s')
        self.logger = logging.getLogger(__file__)
        if targetfile:
            filelogger = logging.FileHandler(filepath)
            filelogger.setFormatter(log_formatter)
            #filelogger.setLevel(logging.INFO)
            self.logger.addHandler(filelogger)
        
        if targetstdout:
            stdoutlogger = logging.StreamHandler(sys.stdout)
            stdoutlogger.setFormatter(log_formatter)
            #stdoutlogger.setLevel(logging.INFO)
            self.logger.addHandler(stdoutlogger)

    def write(self, message):
        if self.log:
            self.logger.error(message)
    
    def flush(self):
        for handler in self.logger.handlers:
            handler.close()

def check_na(df, name='Original'):
    if name:
        print('==== %s =====' % name)
        print(df.shape)
    for k in  df.keys():
        nansize = pd.isnull(df[k]).sum()
        if nansize:
            print('%s - %d' % (k, nansize))
 
def get_projectpath():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
def convert2d(cm, target):
    TP = cm[target, target]
    FP = np.sum(cm[:, target]) - TP
    FN = np.sum(cm[target, :]) - TP
    TN = np.sum(cm) - TP - FP - FN
    #print (TP, FP, FN, TN)
    return np.array([[TP, FP],[FN, TN]])

def plot_score(learners, scores):
    pylab.clf()
    c = 'bcgmrwy'
    width = 0.5
    fig, ax = pylab.subplots()
    rect = ax.bar(np.arange(len(scores)), scores, width, color=c)
    ax.set_xticks(np.arange(len(scores))+width)
    ax.set_xticklabels([l.__class__.__name__ for l in learners])
    
    ax.grid(True)
    ax.set_ylabel('Score')
    ax.set_xlabel('Learner')
    ax.set_title("Learns's Score")
    ax.set_yticks([0, 1.0])
    #pylab.autoscale(tight=True)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%.2f'%height,
                    ha='center', va='bottom')
    autolabel(rect)    
    pylab.show()

def plot_precision_recall(precision, recall, area, title):
    pylab.clf()
    pylab.plot(recall, precision, label='Precision-Recall curve')
    pylab.xlabel('Recall')
    pylab.ylabel('Precision')
    pylab.ylim([0.0, 1.05])
    pylab.xlim([0.0, 1.0])
    pylab.title('Precision-Recall / %s : AUC=%0.2f' % (title, area))
    pylab.legend(loc="lower left")
    pylab.show()

def plot_confusion_matrix(cm, targetname):
    pylab.clf()
    pylab.matshow(cm, fignum=False, cmap='Blues', vmin=0, vmax=1.0)
    ax = pylab.axes()
    ax.set_xticks(range(len(targetname)))
    ax.set_xticklabels(targetname)
    ax.xaxis.set_ticks_position("bottom")
    ax.set_yticks(range(len(targetname)))
    ax.set_yticklabels(targetname)
    pylab.title('Confusion Matrix')
    pylab.colorbar()
    pylab.grid(False)
    pylab.xlabel('Predicted class')
    pylab.ylabel('True class')
    pylab.grid(False)
    pylab.show()
    
def plot_roc(auc_score, tpr, fpr, label=None):
    pylab.clf()
    pylab.figure(num=None, figsize=(5, 4))
    pylab.grid(True)
    pylab.plot([0, 1], [0, 1], 'k--')
    pylab.plot(fpr, tpr)
    pylab.fill_between(fpr, tpr, alpha=0.5)
    pylab.xlim([0.0, 1.0])
    pylab.ylim([0.0, 1.0])
    pylab.xlabel('False Positive Rate')
    pylab.ylabel('True Positive Rate')
    pylab.title('ROC curve (AUC = %0.2f) / %s Vs Rest' %
                (auc_score, label), verticalalignment="bottom")
    pylab.legend(loc="lower right")
    pylab.show()

def persist_cv_file(X, y, cv):
    dataname = 'data'
    suffix = '_cv_%03d.pkl'
    cv_split_filenames = []
    for i, (train, test) in enumerate(cv):
        cv_fold = ([X[k] for k in train], y[train], [X[k] for k in test], y[test])
        cv_split_filename = dataname + suffix % i
        cv_split_filename = os.path.abspath(cv_split_filename)
        joblib.dump(cv_fold, cv_split_filename)
        cv_split_filenames.append(cv_split_filename)
        return cv_split_filenames


def evaluate_cv_file(cv_split_filename, clf, params):
    X_train, y_train, X_test, y_test = joblib.load(
        cv_split_filename, mmap_mode='c')
    
    clf.set_params(**params)
    clf.fit(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    return test_score

def parallel_grid_search(lb_view, clf, cv_split_filenames, param_grid):
    all_tasks = []
    all_parameters = list(IterGrid(param_grid))
    for i, params in enumerate(all_parameters):
        task_for_params = []
        for j, cv_split_filename in enumerate(cv_split_filenames):    
            t = lb_view.apply(evaluate_cv_file,
                              cv_split_filename, clf, params)
            task_for_params.append(t) 
        all_tasks.append(task_for_params)
        
    return all_parameters, all_tasks


if __name__ == '__main__':
    pass