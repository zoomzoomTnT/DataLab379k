# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 09:47:46 2017

@author: Josh M
"""
import numpy as np
import pandas as pd

#data = pd.DataFrame.from_csv('PatientData.csv')

data = pd.read_csv('PatientData.csv', header=None, na_values=["?"])

print(data.values)

print("\n")
print("Problem 5a.")
print("# of Patients " + str(data.shape[0] + 1))
print("# of Features " + str(data.shape[1] + 1))

print("\n")
print("Problem 5b.")
print("Feature 1 : Age in years")
print("Feature 2 : Gender, 0 -> male, 1 -> female")
print("Feature 3 : Weight in pounds")
print("Feature 4 : Height in inches")

print("\n")
print("Problem 5c.")

# Replace missing values in data fram columns
for col in data:
    data = data.replace(np.NaN, data[col].mean())    

df = data.ix[:,12:14]
print(df)

print("\n")
print("Problem 5d.")
print("Iterating through the columns and print a histogram")
print("The columns/features with distributions of higher than avg. variance can be useful")

print("\n")
print("Most Important features: Age, Gender, Patient Condition")