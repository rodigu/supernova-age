import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn

filename = './out/output_1.csv'
two_days = pd.read_csv(filename)
two_days['r-i'] = two_days['BAND_r'] - two_days['BAND_i']
two_days['g-r'] = two_days['BAND_g'] - two_days['BAND_r']
two_days['days_since'] = two_days['MJD'] - two_days['1stDet']

plt.scatter(x = two_days['r-i'],y = two_days['g-r'],c = two_days['days_since'], cmap ='tab10')
plt.ylabel('g-r')
plt.xlabel('r-i')
plt.colorbar(label='days since')
plt.show()
