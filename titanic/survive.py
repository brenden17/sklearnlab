# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
from matplotlib import pyplot as plt 
from sklearnlab.util import check_na
            
def get_fullpath(filename):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), filename))

def get_xy():
    passengers_train = pd.read_csv(get_fullpath('train.csv'), delimiter = ',')
    #print passengers_train.keys()    
    #print passengers_train.head()
    #print passengers_train.describe()
    #check_na(passengers_train)
    X = passengers_train[['Fare', 'Pclass']].values
    y =  passengers_train['Survived'].values
    return X, y

def plot_data(df):
    df.groupby(['Sex','Pclass']).Survived.sum().plot(kind='barh')
    plt.show()

def convert_sex(df):
    df["Sex"] = df["Sex"].apply(lambda sex: 0 if sex == "male" else 1)

def get_arraged_xy():
    passengers_train = pd.read_csv(get_fullpath('train.csv'), delimiter = ',')
    #print passengers_train.describe()
    #check_na(passengers_train)
    #print passengers_train.Age.mean()
    passengers_train.Age = passengers_train.Age.interpolate()
    #print interpolated_passengers.Age.mean()
    #check_na(interpolated_passengers)
    convert_sex(passengers_train)
    X = passengers_train[['Fare', 'Pclass', 'Age', 'Sex']].values
    y = passengers_train['Survived'].values
    return X, y

def create_learner():
    learner =  ExtraTreesClassifier(n_estimators=100, max_depth=None, 
                              min_samples_split=1, random_state=0)
    return [learner, LogisticRegression(), SVC()]
                          
def analysis_cv():
    X, y = get_xy()
    learners = create_learner()
    size = 2
    for learner in learners:
        print '%s - %f' % (learner.__class__.__name__,
                           sum(cross_val_score(learner, X, y, cv=size))/size)
    print '============= After Imputering Age and Sex ============='
    X, y = get_arraged_xy()
    for learner in learners:
        print '%s - %f' % (learner.__class__.__name__,
                           sum(cross_val_score(learner, X, y, cv=size))/size)

if __name__ == '__main__':
    analysis_cv()
    plot_data()
