from sklearn.cluster import SpectralClustering, DBSCAN, OPTICS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def add_axis_subtraction(df):
  if len(df.index) > 100000:
    df = df.sample(n=5000)
  df['r-i'] = df['BAND_r'] - df['BAND_i']
  df['g-r'] = df['BAND_g'] - df['BAND_r']
  df['days_since'] = df['MJD'] - df['1stDet']

  print(df['BAND_r'].isnull().values.any(), df['BAND_i'].isnull().values.any(), df['BAND_g'].isnull().values.any())
  
  df = df[df['days_since'] < 15]
  
  return df

def run_spectral_clustering_2d(df, cluster_num):
  xs = df['r-i']
  ys = df['g-r']
  matrix = np.array([[x, y] for x, y in zip(xs, ys)])
  clustering = SpectralClustering(n_clusters=cluster_num,
    assign_labels='discretize',
    random_state=0).fit(matrix)
  return xs, ys, matrix, clustering.labels_

def plot_clustering_2d(df):

  # plt.figure(1,2,1)
  
  # plt.scatter(x=xs, y=ys, c=clustering, s=30, cmap='tab10')
  plt.axes().set_facecolor("black")
  plt.scatter(x=df['r-i'], y=df['g-r'], c=df['days_since'].astype(int), s=20, cmap='bwr', alpha=.9)
  plt.colorbar()
  plt.xlabel('r-i')
  plt.ylabel('g-r')

def run_spectral_clustering_3d(df, cluster_num):
  xs = df['BAND_r']
  ys = df['BAND_g']
  zs = df['BAND_i']
  matrix = np.array([[x, y, z] for x, y, z in zip(xs, ys, zs)])
  clustering = OPTICS(min_samples=5, cluster_method='dbscan').fit(matrix)
  print(len(set(clustering.labels_)))
  return clustering.labels_

def plot_clustering_3d(df, coloring):
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')

  ax.scatter(df['BAND_r'], df['BAND_g'], df['BAND_i'], c=coloring, alpha=.9, cmap='bwr', s=5)
  ax.set_facecolor("black")

def cluster_df_3d(filename, num_clusters, nrows=5000):
  df = add_axis_subtraction(load_df(filename, nrows))
  clustering = run_spectral_clustering_3d(df, num_clusters)
  new_df = df.copy()
  new_df['cluster'] = clustering
  return new_df

def cluster_df(filename, num_clusters, nrows=5000):
  df = add_axis_subtraction(load_df(filename, nrows))
  plot_clustering_2d(df)
  _, _, _, clustering = run_spectral_clustering_2d(df, num_clusters)
  new_df = df.copy()
  
  new_df['cluster'] = clustering
  return new_df

def write_cluster(df, filename):
  df.to_csv(filename)

def load_df(filename, nrows=5000):
  return pd.read_csv(filename)

if __name__ == '__main__':
  # df = add_axis_subtraction(load_df())
  # xs, ys, matrix, clustering = run_spectral_clustering_2d(df, 10)
  # plot_clustering_2d(xs, ys, matrix, clustering, df)
  # new_df = df.copy()
  # new_df['cluster'] = clustering
  # print(new_df)
  df = cluster_df('./output_1.csv',7)
  write_cluster(df, './out/cluster_df.csv')
  plot_clustering_2d(df)
  df = cluster_df_3d('./output_1.csv', 5, 10000)

  plot_clustering_3d(df, df['days_since'])
  plot_clustering_3d(df, df['cluster'])

  plt.show()
