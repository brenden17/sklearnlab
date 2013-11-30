import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve, auc, roc_curve
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.grid_search import GridSearchCV
from sklearnlab.util import StemmedCountVectorizer, StemmedTfidfVectorizer, \
                            plot_precision_recall, plot_roc
def get_fullpath(filename):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), filename))

def get_xy():
    rawdata = pd.read_csv(get_fullpath('sms_spam.csv'), delimiter = ',')
    le = LabelEncoder()
    y = le.fit_transform(np.array(rawdata['type']))
    vectorizer = StemmedCountVectorizer(min_df=1)
    #vectorizer = StemmedTfidfVectorizer()
    X = vectorizer.fit_transform(np.array(rawdata['text']))
    return X, y

def get_raw_xy():
    rawdata = pd.read_csv(get_fullpath('sms_spam.csv'), delimiter = ',')
    le = LabelEncoder()
    y = le.fit_transform(np.array(rawdata['type']))
    return np.array(rawdata['text']), y

def create_learners():
    return [BernoulliNB(), MultinomialNB(), SVC(), LogisticRegression()]

def create_learner():
    return LogisticRegression()

def create_pipe():
    classer = create_learner()
    vectorizer = StemmedCountVectorizer()
    return Pipeline([('vectorizer', vectorizer), ('clf', classer)])

def analysis_cv():
    X, y = get_xy()
    learners = create_learners()
    size = 2
    for learner in learners:
        print '%s - %f' % (learner.__class__.__name__,
                           sum(cross_val_score(learner, X, y, cv=size))/size)

def analysis():
    X, y = get_xy()    
    X_train, X_test, Y_train, Y_test, = \
                        train_test_split(X.toarray(), y, 
                                         test_size=0.3, random_state=42)
    clf = create_learner()
    clf.fit(X_train, Y_train)
    prob = clf.predict_proba(X_test)[:, 1]
    precision, recall, threld = precision_recall_curve(Y_test, prob)
    
    area = auc(recall, precision)
    plot_precision_recall(precision, recall, area, clf.__class__.__name__)
    ## roc
    fpr, tpr, roc_thresholds = roc_curve(Y_test, prob)
    area = auc(fpr, tpr)
    plot_roc(area, tpr, fpr, clf.__class__.__name__)


def analysis_with_pipeline():
    X, y = get_raw_xy()
    pipe = create_pipe()
    parameters = dict(#vectorizer__ngram_range=[(1,1),(1,2)], 
                        vectorizer__min_df=[1, 2, 4],
                        #vectorizer__stop_words=[None, 'english'],
                        clf__C=[1, 2, 3]
                        )
    grid_search = GridSearchCV(pipe, param_grid=parameters)
    grid_search.fit(X, y)
    print grid_search.best_score_
    print grid_search.best_params_

if __name__ == '__main__':
    #analysis_cv()
    #analysis()
    analysis_with_pipeline()