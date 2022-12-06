from clustering import cluster_df

def cluster_eval(df):
  max_days = 15
  num_clusters = len(df['cluster'].unique())
  step_size = int(max_days/ num_clusters)
  ranges = [{'min': i * step_size, 'max': (i + 1) * step_size} for i in range(num_clusters)]
  cluster_proportions = { cluster: [0] * len(ranges) for cluster in range(num_clusters) }
  print(ranges)
  for ck in cluster_proportions.keys(): # for each cluster
    for idx, r in enumerate(ranges): # for each possible range
      # we check how many that are in this cluster and fit the range
      # print(df[df['cluster'] == ck])
      cluster_proportions[ck][idx] = len(df[(df['cluster'] == ck) & df['days_since'].between(r['min'], r['max'])])
      
    # df['wrong_cluster'] = df.apply(lambda entry: entry in range(ranges[entry['cluster']]['min'], ranges[entry['cluster']]['max']), axis=1)
  return cluster_proportions


def evaluation_to_df(evaluation):
  pass


def plot_cluster_proportions(clusters_evaluations):
  pass


if __name__ == '__main__':
  df = cluster_df(filename='./out/output_1.csv', num_clusters=5)
  print(df)
  evaluation = cluster_eval(df)
  for c in evaluation.keys():
    total = sum(evaluation[c])
    print(f'Cluster: {c}')
    for i, count in enumerate(evaluation[c]):
      print('range  proportion')
      print(f'{i}      {count / total}')