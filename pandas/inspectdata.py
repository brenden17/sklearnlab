# -*- coding: utf-8 -*-
import os
import pandas as pd
from pandas.tools.plotting import scatter_matrix


def get_fullpath(filename):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), filename))

def create_xy():
    rawdata = pd.read_csv(get_fullpath('insurance.csv'), delimiter = ',')
    print('====== describe ======')    
    print rawdata.describe()
    print('====== head ======')    
    print rawdata.head()
    print('====== corr ======')    
    print rawdata.corr()
    print('====== describe ======')    
    scatter_matrix(rawdata)
    rawdata.age.plot()
    rawdata.charges.hist()
    rawdata.boxplot()
    
def arrage_dataframe():
    rdata = pd.read_csv(get_fullpath('insurance.csv'), delimiter = ',')
    sexlabel = rdata.sex.unique()   
    rdata['sexlabel'] = rdata.sex.apply(lambda x: 0 if x==sexlabel[0] else 1)
    

if __name__ == '__main__':
    arrage_dataframe()