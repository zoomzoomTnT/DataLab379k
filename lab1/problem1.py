# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt

mu, sigma = -10, 5
s = np.random.normal(mu, sigma, 1000)

mu2, sigma2 = 10, 5
s2 = np.random.normal(mu2, sigma2, 1000)

s3 = s + s2

count, bins, ignored = plt.hist(s3, 30, normed=True)

plt.show()
