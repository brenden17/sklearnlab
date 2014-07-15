# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from scipy import sparse
from matplotlib import pyplot as plt 
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

def inspect_data(figure=False):
    u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
    users = pd.read_csv(get_fullpath('ml-100k/u.user'),
                                        delimiter = '|', names=u_cols)
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings = pd.read_csv(get_fullpath('ml-100k/u.data'),
                                        delimiter = '\t', names=r_cols)
    m_cols = ['movie_id', 'title', 'release_date', 'video_release_date']    
    movies = pd.read_csv(get_fullpath('ml-100k/u.item'),
                         delimiter = '|', names=m_cols, usecols=range(4))
    
    movie_ratings = pd.merge(movies, ratings)
    user_movie_ratings = pd.merge(movie_ratings, users)
    
    print user_movie_ratings.shape
    
    # movie
    #print user_movie_ratings.groupby('movie_id').size().order(ascending=False)
    #print user_movie_ratings.title.value_counts()[:25]
    #print user_movie_ratings[['age', 'rating']].describe()
    
    # best 15 user rating
    #print ratings.groupby('user_id').size().order(ascending=False)[:15]
    #print ratings.user_id.value_counts()[:15]

    #print user_movie_ratings.occupation.value_counts()[:5]
    #print users.occupation.value_counts()    
    #print users[users.occupation=='programmer'].user_id
    programmer = users[(users.occupation=='programmer') & (users.sex=='F')].user_id
    #print programer
    #print user_movie_ratings[user_movie_ratings.occupation=='programmer'].shape

    programmer_movies = movie_ratings[movie_ratings['user_id'].isin(programmer)]
    programmer_movies.title.value_counts()[:10]
    
    figure=False
    if figure:
        users.age.hist(bins=8)
        plt.show()
        
    
def create_data():
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    data = pd.read_csv(get_fullpath('ml-100k/u.data'),
                                          delimiter = '\t', names=r_cols)
    rating = data.ix[:, 2].values
    user_movie = data.ix[:, 0:2].T.values
    return sparse.csc_matrix((rating, user_movie)).astype(float)
    
def process():
    data = create_data()
    print data[1, :].toarray().shape
    
if __name__ == '__main__':
    inspect_data()
    #process()