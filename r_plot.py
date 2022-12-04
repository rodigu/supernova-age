import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn

filenames = ['./out/output_1.csv','./out/output_2.csv','./out/output_3.csv','./out/output_6.csv']

# two_days = pd.read_csv(filename)
# two_days['r-i'] = two_days['BAND_r'] - two_days['BAND_i']
# two_days['g-r'] = two_days['BAND_g'] - two_days['BAND_r']
# two_days['days_since'] = two_days['MJD'] - two_days['1stDet']
# print(two_days.head(30))
# print(two_days['days_since'].max())
for filename in filenames:
    f = pd.read_csv(filename)
    f['r-i'] = f['BAND_r'] - f['BAND_i']
    f['g-r'] = f['BAND_g'] - f['BAND_r']
    f['days_since'] = f['MJD'] - f['1stDet']
    print(np.sort(f['days_since']))
    # plt.scatter(x = f['r-i'],y = f['g-r'],c = f['days_since'],s = 5, vmax = 60,cmap ='tab10')
    # plt.ylabel('g-r')
    # plt.xlabel('r-i')
    # plt.colorbar(label='days since')
    # plt.title(f'{filename}')
    # plt.show()

