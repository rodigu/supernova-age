from clustering import cluster_df
import pandas as pd
import matplotlib.pyplot as plt

def cluster_eval(df):
  max_days = 15
  num_clusters = len(df['cluster'].unique())
  step_size = int(max_days/ num_clusters)
  ranges = [{'min': i * step_size, 'max': (i + 1) * step_size} for i in range(num_clusters)]
  cluster_proportions = { cluster: [0] * len(ranges) for cluster in range(num_clusters) }
  for ck in cluster_proportions.keys(): # for each cluster
    for idx, r in enumerate(ranges): # for each possible range
      # we check how many that are in this cluster and fit the range
      # print(df[df['cluster'] == ck])
      cluster_proportions[ck][idx] = len(df[(df['cluster'] == ck) & df['days_since'].between(r['min'], r['max'])])
      
    # df['wrong_cluster'] = df.apply(lambda entry: entry in range(ranges[entry['cluster']]['min'], ranges[entry['cluster']]['max']), axis=1)
  
  all_cluster_ratio_list = [cluster_proportions[c] for c in cluster_proportions.keys()]
  cluster_ratio_df = pd.DataFrame(all_cluster_ratio_list,columns=cluster_proportions.keys(),index = ranges)
  return cluster_ratio_df, ranges

def cluster_eval_by_cluster(df):
  max_days = 15
  num_clusters = len(df['cluster'].unique())
  step_size = int(max_days/ num_clusters)
  ranges = [{'min': i * step_size, 'max': (i + 1) * step_size} for i in range(num_clusters)]
  ranges_str = [f"min: i * step_size, max: (i + 1) * step_size" for i in range(num_clusters)]
  cluster_proportions = { cluster: [0] * len(ranges) for cluster in range(num_clusters) }
  print(ranges)
  for ck in cluster_proportions.keys(): # for each cluster
    for idx, r in enumerate(ranges): # for each possible range
      # we check how many that are in this cluster and fit the range
      # print(df[df['cluster'] == ck])
      cluster_proportions[ck][idx] = len(df[(df['cluster'] == ck) & df['days_since'].between(r['min'], r['max'])])
      
    # df['wrong_cluster'] = df.apply(lambda entry: entry in range(ranges[entry['cluster']]['min'], ranges[entry['cluster']]['max']), axis=1)
  
  all_cluster_ratio_list = [cluster_proportions[c] for c in cluster_proportions.keys()]
  cluster_ratio_df = pd.DataFrame(all_cluster_ratio_list,columns=cluster_proportions.keys(),index = ranges)
  return cluster_ratio_df, ranges

def evaluation_to_df(evaluation):
  pass


def plot_cluster_proportions(clusters_evaluations):

  pass


if __name__ == '__main__':
  filename='two_day_range.csv'
  num_clusters=7
  df = cluster_df(filename, num_clusters)
  print(len(df.index))
  cluster_ratio_df, ranges = cluster_eval(df)
  print(cluster_ratio_df)
  plt.imshow(cluster_ratio_df)
  plt.xlabel('cluster number')
  plt.ylabel('day range number')
  plt.colorbar()
  plt.title(f'{num_clusters} Clusters, {filename}')
  plt.show()
