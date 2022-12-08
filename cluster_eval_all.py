from clustering import cluster_df
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def cluster_eval(df):
  max_days = 15
  num_clusters = len(df['cluster'].unique())
  step_size = int(max_days / num_clusters)
  ranges = [{'min': i * step_size, 'max': (i + 1) * step_size} for i in range(num_clusters)]
  print(num_clusters)
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


def hist_plot_cluster_proportions(df):
  bins = range_string(len(df.columns))
  print(bins)
  df.plot(kind='bar', rot=45)
  plt.show()

def to_percent_in_cluster(df):
  new_df = df.copy()
  # for cluster in new_df.columns:
  #   total = new_df[cluster].sum()
  #   for r in new_df.iterrows():
  #     print(r)
  #     df.loc[r][cluster] = df.loc[r][cluster] / total
  return new_df

def cluster_scatter_plot(df,filename):
  plt.scatter(x = df['r-i'], y = df['g-r'], c = df['cluster'])
  return plt.show()

def heatmap_proportions(clusters_evaluations,filename):
  num_clusters = len(clusters_evaluations.columns)
  fig, ax = plt.subplots()
  print(len(clusters_evaluations.index))
  im = ax.imshow(clusters_evaluations)

  # Show all ticks and label them with the respective list entries
  print([n for n in range(0,num_clusters)])
  rows = [int(rng.split(' ')[-1]) for rng in clusters_evaluations.index]
  columns = [n for n in range(0,num_clusters)]
  print(rows, len(rows))
  ax.set_xticks([int(cluster) for cluster in columns], labels= columns)
  ax.set_yticks(range(len(rows)),labels = rows) #[row/num_clusters for row in rows]
  ax.set_ylabel('Ranges')
  ax.set_xlabel('Cluster')
  print(ax.get_xticks(),ax.get_yticks())
  # Rotate the tick labels and set their alignment.
  plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
          rotation_mode="anchor")
  

  ax.set_title(f'Points in Range per Cluster\n{num_clusters} Clusters, {filename}')
  fig.tight_layout()
  return plt.show()

if __name__ == '__main__':
  filename='two_day_range.csv'
  num_clusters=5
  df = cluster_df(filename, num_clusters)
  # print(len(df.index))
  cluster_df, ranges = cluster_eval(df)
  cluster_ratio_df = to_percent_in_cluster(cluster_df)
  print(cluster_ratio_df)
  heatmap_proportions(cluster_ratio_df,filename)
  hist_plot_cluster_proportions(cluster_ratio_df)
  cluster_scatter_plot(df,filename)
  plt.show()
  
