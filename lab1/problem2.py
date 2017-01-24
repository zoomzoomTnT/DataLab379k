# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 10:08:27 2017

@author: Josh M
"""
import numpy as np

from matplotlib import pyplot as plt

bern, p, n = 1, 0.5, 250


z = [None]*1000

for j in range(1000):
    sum = 0
    s = np.random.binomial(bern, p, n)
    for i in range(len(s)):
        s[i] = s[i]*2-1
        sum += s[i]
    avg = sum/n
    z[j] = avg

count, bins, ignored = plt.hist(z, 30, normed=True)
plt.show()