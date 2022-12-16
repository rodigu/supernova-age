from sklearn.cluster import SpectralClustering, Birch, OPTICS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

import os

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

def plot_clustering_plotly(df, zs, color_col):
  # fig = px.scatter_3d(df, x='BAND_r', y='BAND_g', z='BAND_i',
  #             color='cluster')
  fig = go.Figure()
  
  fig.add_trace(go.Scatter3d(x=df['r-i'], y=df['g-r'], z=zs,
            mode='markers',
            marker=dict(color=df[color_col])))
  
  
  fig.show()
  
def birch_cluster_df(dfs: dict[str, pd.DataFrame], vect_columns: list[str], num_clusters: int):
  for sn_type, df in dfs.items():
    _, _, clustering = run_birch_clustering(df, vect_columns, num_clusters)
    new_df = df.copy()
    new_df['cluster'] = clustering
    # print(new_df['cluster'].nunique())
    dfs[sn_type] = new_df
  return dfs

def spectral_cluster_df(dfs: dict[str, pd.DataFrame], vect_columns: list[str], num_clusters: int):
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
    # print(new_df['cluster'].nunique())
    dfs[sn_type] = new_df
  return dfs

def write_cluster(df, filename):
  os.makedirs('/'.join(filename.split('/')[:-1]), exist_ok=True)
  df.to_csv(filename)

def load_df(filename):
  return pd.read_csv(filename).replace([np.inf, -np.inf], np.nan).dropna()

def save_all_clustering():
  dfs = add_axis_subtraction(load_df('./out/output_1_typed.csv'))
  clust_nums = [3,5,7,10]
  out_filenames = ['type_II_cluster.csv','type_Ia_cluster.csv','type_Ibc_cluster.csv']
  sn_types = ['SNIIdf', 'SNIadf', 'SNIbcdf']

  print('Now running spectral clustering')
  save_cluster(dfs, clust_nums, out_filenames, sn_types, 'spectral', spectral_cluster_df)
  print('Now running birch clustering')
  save_cluster(dfs, clust_nums, out_filenames, sn_types, 'birch', birch_cluster_df)

  clust_nums = [10,15,20]
  print('Now running optics clustering')
  save_cluster(dfs, clust_nums, out_filenames, sn_types, 'optics', optics_cluster_df)

def save_cluster(dfs, clust_nums, out_filenames, sn_types, cluster_alg_name, cluster_alg_func):
  for clust_num in clust_nums:
    dfs_typed = cluster_alg_func(dfs, ['BAND_r', 'BAND_i', 'BAND_g'], clust_num)
    for filename, sn_type in zip(out_filenames, sn_types):
      write_cluster(dfs_typed[sn_type], f'./{cluster_alg_name}/band/{clust_num}/' + filename)

    dfs_typed = cluster_alg_func(dfs, ['r-i', 'g-r'], clust_num)
    for filename, sn_type in zip(out_filenames, sn_types):
      write_cluster(dfs_typed[sn_type], f'./{cluster_alg_name}/diff/{clust_num}/' + filename)

def plot_3d_clustered():
  filenames = ['./birch/band/7/type_II_cluster.csv','./birch/band/7/type_Ia_cluster.csv','./birch/band/7/type_Ibc_cluster.csv']
  dfs = [load_df(file) for file in filenames]
  for df in dfs:
    plot_clustering_plotly(df, df['cluster'], 'days_since')

def further_clustering_analysis():
  sn_types = ['SNIIdf', 'SNIadf', 'SNIbcdf']
  filenames = ['./birch/band/5/type_II_cluster.csv','./birch/band/5/type_Ia_cluster.csv','./birch/band/5/type_Ibc_cluster.csv']
  dfs = {sn_type: load_df(file) for sn_type, file in zip(sn_types, filenames)}
  for df in dfs.values():
    df['cluster_diff'] = -1
    for cluster_id in range(df['cluster'].nunique()):
      _, _, clustering = run_spectral_clustering(df[df['cluster'] == cluster_id], cluster_num=5, vect_columns=['BAND_i', 'BAND_r', 'BAND_g'])
      df.loc[df['cluster'] == cluster_id, 'cluster_diff'] = clustering
  
  for df in dfs.values():
    plot_clustering_plotly(df, df['cluster'], 'cluster_diff')

  plot_3d_clustered()

if __name__ == '__main__':
  dfs = add_axis_subtraction(load_df('./out/output_1_typed.csv'))
  save_all_clustering()
