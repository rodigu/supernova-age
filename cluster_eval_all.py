from clustering import cluster_df

def cluster_eval(df):
  max_days = 15
  num_clusters = len(df['cluster'].unique())
  step_size = num_clusters / max_days
  ranges = [{'min': i * step_size, 'max': (i + 1) * step_size} for i in range(num_clusters)]
  df['wrong_cluster'] = df.apply(lambda entry: entry['days_since'] in range(ranges[entry['cluster']]['min'], ranges[entry['cluster']]['max']))
  
  total_count = len(df)
  total_incorrect = len(df[~df['wrong_cluster']])

  return total_incorrect / total_count


if __name__ == '__main__':
  df = cluster_df()
  cluster_eval(df)