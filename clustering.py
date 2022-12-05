from sklearn.cluster import SpectralClustering
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def add_axis_subtraction(df):
  df['r-i'] = df['BAND_r'] - df['BAND_i']
  df['g-r'] = df['BAND_g'] - df['BAND_r']
  df['days_since'] = df['MJD'] - df['1stDet']
  
  df = df[df['days_since'] < 10]
  return df

def run_spectral_clustering_2d(df, cluster_num):
  xs = df['r-i']
  ys = df['g-r']
  matrix = np.array([[x, y] for x, y in zip(xs, ys)])
  clustering = SpectralClustering(n_clusters=cluster_num,
    assign_labels='discretize',
    random_state=0).fit(matrix)
  return xs, ys, matrix, clustering.labels_

def plot_clustering_2d(xs, ys, matrix, clustering, df):

  print(len(xs), len(df['days_since']))
  # plt.figure(1,2,1)
  
  # plt.scatter(x=xs, y=ys, c=clustering, s=30, cmap='tab10')
  plt.scatter(x=xs, y=ys, c=df['days_since'].astype(int), s=20, cmap='tab10', alpha = .5)

  

def load_df(filename='./out/output_1.csv'):
  return pd.read_csv(filename)

if __name__ == '__main__':
  df = add_axis_subtraction(load_df())
  xs, ys, matrix, clustering = run_spectral_clustering_2d(df, 10)
  plot_clustering_2d(xs, ys, matrix, clustering, df)