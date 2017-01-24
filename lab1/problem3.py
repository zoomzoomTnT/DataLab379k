# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 10:42:36 2017

@author: Josh M
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# Mean Calculation
def mean(n):
    return float(sum(n))/max(len(n),1)

# Standard Deviation
def std(n):
    return math.sqrt(mean(abs(n-mean(n))**2))

# Gaussian Distribution    
mu, sigma = 0, 5
s = np.random.normal(mu, sigma, 25000)

# Mean of Gaussuan, should be ~0
x = mean(s)
print("\n")
print("Mean:  "+ str(x))

# Standard Deviation of Gaussian Dist.
y = std(s)
print("\n")
print("Standard Deviation:  " + str(y))

plt.show()
