import numpy as np
from clustering import cluster_df

def per_cluster_ratio(df):
    clusters = df['cluster'].unique()
    number_of_clusters = len(list(clusters))
    step_size = int(15/number_of_clusters)
    cluster_ratios = []
    for i in clusters:
        df_cluster = df[df['cluster']==i]
        ranges = [(i*step_size,(i+1)*step_size) for i in range(number_of_clusters)]
        len_of_cluster = len(df_cluster)
        cluster_correct = 0
        for _, sn in df[df['cluster']==i].iterrows():
            if sn['days_since'] in range(ranges[i][0], ranges[i][1]):
                cluster_correct += 1
        cluster_ratios.append(cluster_correct/len_of_cluster)
    return cluster_ratios

if __name__ == '__main__':
    df = cluster_df()
    print(per_cluster_ratio(df))