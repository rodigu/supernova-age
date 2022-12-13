from sklearn.cluster import SpectralClustering, DBSCAN, OPTICS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def add_axis_subtraction(df: pd.DataFrame, sample_size=5000, max_age=15) -> dict[str, pd.DataFrame]:
  """Adds axis subtraction (r-i and g-r) and supernova age to given dataframe.
  Filters out supernovae that are older than 15 days.
  Separates supernova by type.
  Samples can be extracted if dataset is too large

  :param df: dataframe with r, g and i bands, MJD and 1stDet date
  :param sample_size: size of sample to be extracted from dataframe, defaults to 5000
  :param max_age: max age of a supernova, defaults to 15
  :return: dictionary keyed by supernova types with their respective dataframes as values
  """
  df = df.sample(n=sample_size)
  
  df['days_since'] = df['MJD'] - df['1stDet']
  df = df[df['days_since'] < max_age]

  df['r-i'] = df['BAND_r'] - df['BAND_i']
  df['g-r'] = df['BAND_g'] - df['BAND_r']
  
  SNIIdf = df[df['parsnip_type']==1]
  SNIadf = df[df['parsnip_type']==0]
  SNIbcdf = df[df['parsnip_type']==2]
  # print(df[df['r-i'].isnull()])
  
  return {'SNIIdf': SNIIdf, 'SNIadf': SNIadf, 'SNIbcdf': SNIbcdf}

def run_spectral_clustering(df: pd.DataFrame, cluster_num: int, vect_columns: list[str]) -> tuple[list[pd.Series], np.array, list[int]]:
  """Runs spectral clustering on given dataframe

  :param df: dataframe with supernova information
  :param cluster_num: number of clusters
  :param vect_columns: which columns of the dataframe to use as vectors for the clustering
  :return: list with vectors as pd.Series, matrix [vectors], clustering labels list
  """
  vectors, matrix = generate_matrix(df, vect_columns)
  clustering = SpectralClustering(n_clusters=cluster_num,
    assign_labels='discretize',
    random_state=0).fit(matrix)
  return vectors, matrix, clustering.labels_

def generate_matrix(df: pd.DataFrame, vect_columns: list[str]) -> tuple[list[pd.Series], np.array]:
  """Generates matrix from given df and vector columns

  :param df: 
  :param vect_columns: columns to be used as vectors
  :return: list with vectors and matrix
  """
  vectors = []
  for v in vect_columns:
    vectors.push(df[v])
  return vectors, np.array(vectors).T

def run_optics_clustering(df: pd.DataFrame, vect_columns: list[str]) -> tuple[list[pd.Series], np.array, list[int]]:
  vectors, matrix = generate_matrix(df, vect_columns)
  clustering = OPTICS(min_samples=5, cluster_method='dbscan').fit(matrix)
  print(len(set(clustering.labels_)))
  return vectors, matrix, clustering.labels_

def run_spectral_clustering_2d(df: pd.DataFrame, cluster_num: int, by_x='r-i', by_y='g-r') -> tuple[pd.Series, pd.Series, np.array, list[int]]:
  """Runs spectral clustering on given dataframe

  :param df: dataframe with supernova information
  :param cluster_num: number of clusters
  :param by_x: which column to use as an x vector for the clustering algorithm
  :param by_y: which column to use as a y vector for the clustering algorithm
  :return: x vector, y vector, matrix [x,y], clustering labels list
  """
  xs = df[by_x]
  ys = df[by_y]
  matrix = np.array([xs, ys]).T
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
  matrix = np.array([xs, ys, zs]).T
  clustering = OPTICS(min_samples=5, cluster_method='dbscan').fit(matrix)
  print(len(set(clustering.labels_)))
  return clustering.labels_



def plot_clustering_3d(df, coloring):
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')

  ax.scatter(df['BAND_r'], df['BAND_g'], df['BAND_i'], c=coloring, alpha=.9, cmap='bwr', s=5)
  ax.set_facecolor("black")

def cluster_df_3d(filename, num_clusters, nrows=5000):
  dfs = add_axis_subtraction(load_df(filename, nrows))
  for sn_type, df in dfs.items():
    clustering = run_spectral_clustering_3d(df, num_clusters)
    new_df = df.copy()
    new_df['cluster'] = clustering
    dfs[sn_type] = new_df
  return dfs

def cluster_df(filename, num_clusters, nrows=5000):
  dfs = add_axis_subtraction(load_df(filename, nrows))
  for sn_type, df in dfs.items():
    plot_clustering_2d(df)
    _, _, _, clustering = run_spectral_clustering_2d(df, num_clusters)
    new_df = df.copy()
    
    new_df['cluster'] = clustering
    dfs[sn_type] = new_df
  return dfs

def write_cluster(df, filename):
  df.to_csv(filename)

def load_df(filename, nrows=5000):
  return pd.read_csv(filename).replace([np.inf, -np.inf], np.nan).dropna()

if __name__ == '__main__':
  # df = add_axis_subtraction(load_df())
  # xs, ys, matrix, clustering = run_spectral_clustering_2d(df, 10)
  # plot_clustering_2d(xs, ys, matrix, clustering, df)
  # new_df = df.copy()
  # new_df['cluster'] = clustering
  # print(new_df)
  dfs_typed = cluster_df('./output_1_typed.csv',7)
  for sn_type, df in dfs_typed.items():
    # write_cluster(df, './out/cluster_df.csv')
    plot_clustering_2d(df)

  
  dfs_typed = cluster_df_3d('./output_1_typed.csv', 5, 10000)
  for sn_type, df in dfs_typed.items():
    plot_clustering_3d(df, df['days_since'])
    plot_clustering_3d(df, df['cluster'])

  plt.show()
