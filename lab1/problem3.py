# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 10:42:36 2017

@author: Josh M
"""

import numpy as np
import matplotlib.pyplot as plt

mu, sigma = 0, 5
s = np.random.normal(mu, sigma, 25000)

plt.hist(s, 30, normed=True)

x = np.mean(s)
y = np.std(s)
print("mean: "+ str(x) + " sd: " + str(y))
plt.show()
