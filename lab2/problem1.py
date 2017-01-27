# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 09:42:21 2017

@author: Josh M
"""

import numpy as np
from pandas import *
#from libraries.settings import *
from scipy.stats.stats import pearsonr
import itertools

df = pd.read_csv('DF1.csv', usecols=[1,2,3,4])

pairwisecorr = {}
df.columns = ['col1', 'col2', 'col3', 'col4']
col = df.columns.tolist()

print(df)

for col_a, col_b in itertools.combinations(col, 2):
    pairwisecorr[col_a + '__' + col_b] = pearsonr(df.loc[:, col_a], df.loc[:, col_b])

result = DataFrame.from_dict(pairwisecorr, orient='index')
result.columns = ['PCC', 'p-value']

print(result.sort_index())