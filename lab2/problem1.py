# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 09:42:21 2017

@author: Josh M

"""
import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sns

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

result = pd.DataFrame.from_dict(pairwisecorr, orient='index')
result.columns = ['pcc', 'pvalue']

print(result.ix[:,0:1])

plt.scatter(df['col1'], df['col3'])
plt.show()

plt.scatter(df['col2'], df['col4'])
plt.show()


