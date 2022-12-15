import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

def cluster_eval(df):
  max_days = 15
  num_clusters = len(df['cluster'].unique())
  step_size = max_days / num_clusters
  ranges = [{'min': i * step_size, 'max': (i + 1) * step_size} for i in range(num_clusters)]
  # print(num_clusters)
  cluster_proportions = { cluster: [0] * len(ranges) for cluster in range(num_clusters) }
  for idx, r in enumerate(ranges):
    for cluster in cluster_proportions.keys(): # for each cluster
      # print(cluster, r)
      is_cluster = df['cluster'] == cluster
      is_in_range = df['days_since'].between(r['min'], r['max'])
      cluster_proportions[cluster][idx] = len(df[is_cluster & is_in_range])
      # print(len(df[is_cluster & is_in_range]))

  all_cluster_ratio_list = np.array([cluster_proportions[c] for c in cluster_proportions.keys()]).T
  # print(all_cluster_ratio_list)
  cluster_ratio_df = pd.DataFrame(all_cluster_ratio_list,columns=cluster_proportions.keys(),index = range_string(num_clusters))
  return cluster_ratio_df, ranges


def range_string(num_clusters):
  max_days = 15
  step_size = max_days/ num_clusters
  return [f"{i * step_size:2f} to {(i + 1) * step_size:2f}" for i in range(num_clusters)]


def hist_plot_cluster_proportions(in_df,outfilename, filename):
  df = in_df.transpose()
  df.plot(kind='bar', rot=45, colormap='viridis', title = f"{filename.split('.')[0].replace('_',' ')} Days Since per Cluster")
  plt.savefig(outfilename)
  # else:
  # return sns.histplot(df, x='days_since', hue="cluster", element="step", stat="density", common_norm=False)
  

def to_percent_in_cluster(df):
  new_df = df.copy()
  # for cluster in new_df.columns:
  #   total = new_df[cluster].sum()
  #   for r in new_df.iterrows():
  #     print(r)
  #     df.loc[r][cluster] = df.loc[r][cluster] / total
  return new_df
def days_since_scatter_plot(df,filename,outfilename):
  plt.scatter(x = df['r-i'], y = df['g-r'], c = df['days_since'])
  plt.xlabel('r-i')
  plt.ylabel('g-r')
  plt.title(f"{filename.split('.')[0].replace('_', ' ')} Days Since First Detection Scatterplot")
  plt.colorbar()
  plt.savefig(outfilename)
def cluster_scatter_plot(df,filename,outfilename):
  ndf = df[df['cluster'] >= 0]
  plt.scatter(x = ndf['r-i'], y = ndf['g-r'], c = ndf['cluster'], cmap = 'tab20')
  plt.xlabel('r-i')
  plt.ylabel('g-r')
  plt.colorbar()
  plt.title(f"{filename.split('.')[0].replace('_', ' ')} Cluster Scatterplot")
  plt.savefig(outfilename)
  

def heatmap_proportions(clusters_evaluations,filename:str,outfilename):
  num_clusters = len(clusters_evaluations.columns)
  # fig, ax = plt.subplots()
  # print(len(clusters_evaluations.index))
  # im = ax.imshow(clusters_evaluations)
  htmp = sns.heatmap(clusters_evaluations, annot=True,  linewidths=.5,fmt = 'd')
  htmp.set(xlabel ="Cluster", ylabel = "Range", title =filename.split('.')[0].replace('_', ' '))
  fig = htmp.get_figure()
  fig.savefig(outfilename,bbox_inches='tight')

if __name__ == '__main__':
  foldernames=['optics_banddiff_20', 'optics_band_20','optics_banddiff_15', 'optics_band_15', 'optics_banddiff_10', 'optics_band_10','spectral_band_3','spectral_banddiff_3', 'spectral_band_5','spectral_banddiff_5', 'spectral_band_7','spectral_banddiff_7','spectral_band_10','spectral_banddiff_10']
  filenames=['type_II_cluster.csv','type_Ia_cluster.csv','type_Ibc_cluster.csv']
  for folder in foldernames:
    for filename in filenames:
      df = pd.read_csv('./' + folder + '/' + filename)
      # dfs = cluster_df(filename, num_clusters, 20000)
      # for df in dfs:

      actual_cluster_num = len(df['cluster'].unique())
      # print(df[df['cluster'] < 0])

      # df['cluster'] = df['cluster'] + 1
      # # print(len(df.index))
      cluster_df, ranges = cluster_eval(df)
      # print(df['cluster'].unique())
      # print(df[df['days_since'] <= 3])
      print(cluster_df)
      plt.figure()
      heatmap_proportions(cluster_df,filename, './' + folder + '/' + 'heatmap_' + filename.replace('.csv', '.png'))

      # plt.figure()
      hist_plot_cluster_proportions(cluster_df, './' + folder + '/' + 'hist_' + filename.replace('.csv', '.png'), filename)
      # cluster_scatter_plot(df,filename)
      # plt.savefig('./' + folder + '/' + 'misc_' + filename.replace('.csv', '.png'))
      plt.figure()
      days_since_scatter_plot(df,filename, './' + folder + '/' + 'days_since_scatter_' + filename.replace('.csv', '.png'))
      plt.figure()
      cluster_scatter_plot(df,filename, './' + folder + '/' + 'cluster_scatter_' + filename.replace('.csv', '.png'))