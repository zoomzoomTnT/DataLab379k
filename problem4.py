# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 11:15:35 2017

@author: Josh M
"""
import numpy as np
import matplotlib.pyplot as plt

mean = [-5,5]
cov = [[20,.8], [.8,30]]

x,y = np.random.multivariate_normal(mean, cov, 10000).T
plt.plot(x,y,'x')
plt.axis('equal')

print(np.cov(x,y))
print(np.mean(np.mean(x)+np.mean(y)))

plt.show()