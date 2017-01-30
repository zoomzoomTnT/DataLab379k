import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
df = pd.read_csv('DF2',usecols=[1,2])
df.columns = ['col1','col2']
plt.scatter(df['col1'],df['col2'])
plt.show()
