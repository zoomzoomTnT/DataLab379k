# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt

# Mean Calculation
def mean(n):
    return float(sum(n))/max(len(n),1)

mu, sigma = -10, 5
s = np.random.normal(mu, sigma, 1000)

mu2, sigma2 = 10, 5
s2 = np.random.normal(mu2, sigma2, 1000)

s3 = s + s2
print("Problem 1a: The Sum of two gaussians is gaussian")
print("\n")

count, bins, ignored = plt.hist(s3, 30, normed=True)
print("Problem 1b: Estimated Mean : 0, Estimated Varience : 5")

plt.show()
