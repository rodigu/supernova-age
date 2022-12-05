import numpy as np

def per_cluster_ratio(df):
    clusters = df['cluster'].unique()
    number_of_clusters = len(list(clusters))
    step_size = 15/number_of_clusters
    for i in clusters:
        df_cluster = df[df['cluster']==i]
        ranges = [(i*step_size,(i+1)*step_size) for i in range(number_of_clusters)]
        len_of_cluster = len(df_cluster)
        cluster_correct = 0
        for sn in df[df['cluster']==i]:
            if df['days_since'][sn] in range(ranges[i][0], ranges[i][1]):
                cluster_correct += 1
        ratio = (cluster_correct/len_of_cluster)