# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from sklearn_pandas import DataFrameMapper
from sklearn import preprocessing
from matplotlib import pyplot as plt 

from sklearnlab.util import convert2d, plot_precision_recall,\
                            plot_confusion_matrix, plot_roc,\
                            plot_score, Logger, check_na

def get_fullpath(filename):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), filename))

def create_data(figure=False):
    snsdata = pd.read_csv(get_fullpath('snsdata.csv'), delimiter = ',')
    #print snsdata.describe()
    #print snsdata.age.value_counts()
    #print snsdata.age.describe()
    #snsdata.boxplot()
    #plt.show()
    # check nan
    check_na(snsdata, name='Original')
    # drop all nan
    excluded_nan_snsdata =  snsdata.dropna()
    check_na(excluded_nan_snsdata, 'excluded_nan_snsdata')      
    # interpolate
    print '===== interpolate ====='
    interpolated_snsdata = snsdata.copy()
    print interpolated_snsdata.age.describe()
    interpolated_snsdata.age =  interpolated_snsdata.age.interpolate()
    print interpolated_snsdata.age.describe()    
    check_na(interpolated_snsdata, 'interpolate_nan_snsdata')
    print '===== apply ====='
    print snsdata.age.describe()
    applied_snsdata = snsdata.copy()
    applied_snsdata.age = applied_snsdata.age.apply(lambda x: x if x>=13 and x<20 else np.NaN)
    print applied_snsdata.age.describe()
    print '===== created new features ====='
    check_na(snsdata, name='Original')
    created_feature_snsdata = snsdata.copy()
    created_feature_snsdata['gender_no'] = \
                            created_feature_snsdata.gender.apply(
                                    lambda x: 'F' if type(x)==float and np.isnan(x) else x)
    print '===== aggregate ====='
    aggregated_snsdata = snsdata.copy()
    gradyear_snsdata = aggregated_snsdata.groupby('gradyear')
    print gradyear_snsdata.age.aggregate(np.mean)
    gender_snsdata = aggregated_snsdata.groupby('gender')
    print gender_snsdata.age.aggregate(np.mean)
    
    print '===== Normalized and Standardize ====='
    new_column =['gradyear', 'age', 'friends', 'gender','basketball',
         'football', 'soccer', 'softball', 'volleyball', 'swimming',
         'cheerleading', 'baseball', 'tennis', 'sports', 'cute',
         'sex', 'sexy', 'hot', 'kissed', 'dance',
         'band', 'marching', 'music', 'rock', 'god',
         'church', 'jesus', 'bible', 'hair', 'dress',
         'blonde', 'mall', 'shopping', 'clothes', 'hollister',
         'abercrombie', 'die', 'death', 'drunk', 'drugs']
    new_column_snsdata = snsdata.reindex_axis(new_column, axis=1)
    fixed_snsdata = new_column_snsdata.ix[:, :3]
    print fixed_snsdata.shape
    last_snsdata = new_column_snsdata.ix[:, 3:]
    standardized_factor = [(k, preprocessing.StandardScaler()) for k in snsdata.keys()[4:]]
    standardized_factor.insert(0, ('gender', preprocessing.LabelEncoder()))
    mapper = DataFrameMapper(standardized_factor)
    xx = mapper.fit_transform(last_snsdata)
    print fixed_snsdata.values.T[:, 0]
    print fixed_snsdata.values.T.shape
    s = np.hstack((fixed_snsdata.values, xx))
    print s.shape
    print s[:, 1]
    
    print '===== the best way ====='
    print snsdata.shape
    df = snsdata.copy()    
    basic_info = df[['gradyear', 'age', 'friends', 'gender']]
    print basic_info.shape
    print basic_info.ix[:, :2]
    features = df[snsdata.keys()[4:]]
    print features.shape
    print features.mean() #features.describe()
    scaler = preprocessing.StandardScaler()
    scaled_features = scaler.fit_transform(features.values)
    scaled_df = pd.DataFrame(scaled_features)
    print scaled_df.shape    
    print scaled_df.mean()
    s = np.hstack((basic_info.values, scaled_features))
    print s.shape
    print s[0, :]
    print s[1, :]
    
if __name__ == '__main__':
    create_data()
