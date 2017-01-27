# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 09:42:21 2017

@author: Josh M
"""

import numpy as np
import pandas as pd

df = pd.read_csv('DF1.csv', usecols=[1,2,3,4])

df.columns = ['col1', 'col2', 'col3', 'col4']

print(df)
