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
  step_size = int(max_days/ num_clusters)
  return [f"{i * step_size} to {(i + 1) * step_size}" for i in range(num_clusters)]


def hist_plot_cluster_proportions(in_df):
  df = in_df.transpose()
  mpl_plt = df.plot(kind='bar', rot=45, colormap='viridis')
  return mpl_plt
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

def cluster_scatter_plot(df,filename):
  fig, (ax1, ax2) = plt.subplots(1, 2)
  fig.suptitle(filename)
  ax1.scatter(x = df['r-i'], y = df['g-r'], c = df['days_since'])
  # divider = make_axes_locatable(ax1)
  # cax1 = divider.append_axes('left', size='5%', pad=0.05)
  ax2.scatter(x = df['r-i'], y = df['g-r'], c = df['cluster'])
  # divider = make_axes_locatable(ax2)
  # cax2 = divider.append_axes('right', size='5%', pad=0.05)
  # fig.colorbar(ax1,cax=cax1,orientation='vertical')
  # fig.colorbar(ax2,cax=cax2,orientation='vertical')
  # ax3.hist_plot_cluster_proportions(df)
  # ax4 = sns.heatmap(clusters_evaluations, annot=True,  linewidths=.5,fmt = 'd')
  # ax4.set(xlabel ="Cluster", ylabel = "Range", title =filename)
  # sns.jointplot(data=df, x='r-i', y='g-r',hue = 'days_since')
  
  return plt.show()

def heatmap_proportions(clusters_evaluations,filename):
  num_clusters = len(clusters_evaluations.columns)
  # fig, ax = plt.subplots()
  # print(len(clusters_evaluations.index))
  # im = ax.imshow(clusters_evaluations)
  htmp = sns.heatmap(clusters_evaluations, annot=True,  linewidths=.5,fmt = 'd')
  htmp.set(xlabel ="Cluster", ylabel = "Range", title =filename)
  return htmp

if __name__ == '__main__':
  filenames=['./optics_band_5_type_II_cluster.csv','./optics_band_5_type_Ia_cluster.csv','./optics_band_5_type_Ibc_cluster.csv']
  for filename in filenames:
    df = pd.read_csv(filename)
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
    cluster_scatter_plot(df,filename)
    heatmap_proportions(cluster_df,filename)
    hist_plot_cluster_proportions(cluster_df)

    # cluster_scatter_plot(df,filename)
    plt.show()
