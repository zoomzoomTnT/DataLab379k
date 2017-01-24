# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 11:15:35 2017

@author: Josh M, Zhicong Z
"""

import numpy as np

# Mean Calculation
def mean(n):
    return float(sum(n))/max(len(n),1)

# Covarience Calculation
def cov(i,j):
    i_m = mean(i)
    j_m = mean(j)
    count = 0
    
    for index in range(0, len(i)):
        # Local cov, joint.
        lc = ((i[index] - i_m) * (j[index] - j_m))
        count += lc
    
    ret = count/(len(i) - 1)
    return ret   
    
# Mult. Gaussian Dist. Parameters
mltmean = [-5,5]
mltcov = [[20,.8], [.8,30]]

# Multivariate Gauss. Distribution
x,y = np.random.multivariate_normal(mltmean, mltcov, 10000).T

# Covarience Matrix for x,y
joint_xx = np.vstack((x,x))
joint_xy = np.vstack((x,y))
joint_yx = np.vstack((y,x))
joint_yy = np.vstack((y,y))
print("\n")
print("Covarience Matrix:")
print("[" + str(cov(x,x)) + " , " + str(cov(x,y)) + "]")
print("[" + str(cov(y,x)) + " , " + str(cov(y,y)) + "]")

# Mean x,y
print("\n")
avg = (mean(x) + mean(y)) / 2
print("Mean: " + str(avg))