# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 11:15:35 2017

@author: Josh M
"""

import numpy as np
import matplotlib.pyplot as plt

def mean(n):
    return float(sum(n))/max(len(n),1)

mltmean = [-5,5]
mltcov = [[20,.8], [.8,30]]

x,y = np.random.multivariate_normal(mltmean, mltcov, 10000).T
print("\n")
print("Covarience Matrix:")
print(np.cov(x,y))

print("\n")
avg = (mean(x) + mean(y)) / 2
print("Mean: " + str(avg))

plt.show()