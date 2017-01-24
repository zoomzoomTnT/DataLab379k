# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 10:42:36 2017

@author: Josh M
"""

import math
import numpy as np
import matplotlib.pyplot as plt


def mean(n):
    return float(sum(n))/max(len(n),1)

mu, sigma = 0, 5
s = np.random.normal(mu, sigma, 25000)

x = mean(s)
y = math.sqrt(mean(abs(s - mean(s))**2))
print("\n")
print("Mean:  "+ str(x))

print("\n")
print("Standard Deviation:  " + str(y))

plt.show()
