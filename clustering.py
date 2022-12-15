from sklearn.cluster import SpectralClustering, Birch, OPTICS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def add_axis_subtraction(df: pd.DataFrame, sample_size=15000, max_age=15) -> dict[str, pd.DataFrame]:
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
    vectors.append(df[v])
  return vectors, np.array(vectors).T

def run_birch_clustering(df: pd.DataFrame, vect_columns: list[str], n_clusters=5):
  vectors, matrix = generate_matrix(df, vect_columns)
  clustering = Birch(n_clusters=n_clusters, threshold=.06).fit(matrix)
  return vectors, matrix, clustering.labels_

def run_optics_clustering(df: pd.DataFrame, vect_columns: list[str], min_samples=5) -> tuple[list[pd.Series], np.array, list[int]]:
  vectors, matrix = generate_matrix(df, vect_columns)
  clustering = OPTICS(min_samples=min_samples).fit(matrix)
  return vectors, matrix, clustering.labels_

def plot_clustering_2d(df: pd.DataFrame, title:str, coloring='days_since', columns=['r-i', 'g-r']):

  # plt.figure(1,2,1)
  
  # plt.scatter(x=xs, y=ys, c=clustering, s=30, cmap='tab10')
  plt.axes().set_facecolor("black")
  plt.title(title)
  plt.scatter(x=df[columns[0]], y=df[columns[1]], c=df[coloring].astype(int), s=20, cmap='bwr', alpha=.9)
  plt.colorbar()
  plt.xlabel('r-i')
  plt.ylabel('g-r')

def plot_clustering_3d(df: pd.DataFrame, title:str, coloring='days_since', columns=['BAND_r','BAND_g','BAND_i']):
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')

  ax.scatter(df[columns[0]], df[columns[1]], df[columns[2]], c=df[coloring], alpha=.9, cmap='bwr', s=5)
  ax.set_title(title)
  ax.set_facecolor("black")

def birch_cluster_df(dfs: dict[str, pd.DataFrame], num_clusters: int, vect_columns: list[str]):
  for sn_type, df in dfs.items():
    _, _, clustering = run_birch_clustering(df, vect_columns, num_clusters)
    new_df = df.copy()
    new_df['cluster'] = clustering
    print(new_df['cluster'].nunique())
    dfs[sn_type] = new_df
  return dfs

def spectral_cluster_df(dfs: dict[str, pd.DataFrame], num_clusters: int, vect_columns: list[str]):
  for sn_type, df in dfs.items():
    _, _, clustering = run_spectral_clustering(df, num_clusters, vect_columns)
    new_df = df.copy()
    new_df['cluster'] = clustering
    dfs[sn_type] = new_df
  return dfs

def optics_cluster_df(dfs: dict[str, pd.DataFrame], vect_columns:list[str], min_samples:int):
  for sn_type, df in dfs.items():
    _, _, clustering = run_optics_clustering(df, vect_columns, min_samples)
    new_df = df.copy()
    
    new_df['cluster'] = clustering
    print(new_df['cluster'].nunique())
    dfs[sn_type] = new_df
  return dfs

def write_cluster(df, filename):
  df.to_csv(filename)

def load_df(filename):
  return pd.read_csv(filename).replace([np.inf, -np.inf], np.nan).dropna()

def save_clustering_out():
  dfs = add_axis_subtraction(load_df('./output_1_typed.csv'))
  clust_nums = [3,5,7,10]
  out_filenames = ['type_II_cluster.csv','type_Ia_cluster.csv','type_Ibc_cluster.csv']
  sn_types = ['SNIIdf', 'SNIadf', 'SNIbcdf']

  for clust_num in clust_nums:
    dfs_typed = spectral_cluster_df(dfs, clust_num, ['r-i', 'g-r'])
    for filename, sn_type in zip(out_filenames, sn_types):
      write_cluster(dfs_typed[sn_type], f'./spectral_banddiff_{clust_num}/' + filename)

    dfs_typed = spectral_cluster_df(dfs, clust_num, ['BAND_r', 'BAND_i', 'BAND_g'])
    for filename, sn_type in zip(out_filenames, sn_types):
      write_cluster(dfs_typed[sn_type], f'./spectral_band_{clust_num}/' + filename)

  clust_nums = [10,15,20]
  for clust_num in clust_nums:
    dfs_typed = optics_cluster_df(dfs, ['BAND_r', 'BAND_i', 'BAND_g'], clust_num)
    for filename, sn_type in zip(out_filenames, sn_types):
      write_cluster(dfs_typed[sn_type], f'./optics_band_{clust_num}/' + filename)

    dfs_typed = optics_cluster_df(dfs, ['r-i', 'g-r'], clust_num)
    for filename, sn_type in zip(out_filenames, sn_types):
      write_cluster(dfs_typed[sn_type], f'./optics_banddiff_{clust_num}/' + filename)

if __name__ == '__main__':
  dfs = add_axis_subtraction(load_df('./output_1_typed.csv'))
  
  dfs_typed = birch_cluster_df(dfs, 5, ['r-i', 'g-r'])

  # for sn_type, df in dfs_typed.items():
  #   print(df['cluster'])
  #   plot_clustering_3d(df, 'days since', 'days_since')
  #   plot_clustering_3d(df, 'cluster', 'cluster')
  plot_clustering_2d(dfs_typed['SNIIdf'], 'Type Ia, days since', coloring='cluster')
  plt.show()
  
  plot_clustering_2d(dfs_typed['SNIadf'], 'Type Ia, days since', coloring='cluster')
  plt.show()

  plot_clustering_2d(dfs_typed['SNIbcdf'], 'Type Ibc, clustering', coloring='cluster')
  plt.show()

  # save_clustering_out()
