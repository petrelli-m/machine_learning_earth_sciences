import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, set_link_color_palette

def plot_dendrogram(model, **kwargs):
    
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count +=1
            else:
                current_count += counts[child_idx-n_samples]
        counts[i] = current_count
        
    linkage_matrix = np.column_stack([model.children_, 
                                      model.distances_,
                                     counts]).astype(float)
    
    dendrogram(linkage_matrix, **kwargs)

model = AgglomerativeClustering(linkage='ward',  
                                affinity='euclidean',
                                distance_threshold = 0, 
                                n_clusters=None)

model.fit(my_dataset_ilr_scaled)
    
fig, ax = plt.subplots(figsize = (10,6))
ax.set_title('Hierarchical clustering dendrogram')

plot_dendrogram(model, truncate_mode='level', p=5, 
                color_threshold=0, 
                above_threshold_color='black')

ax.set_xlabel('Number of points in node')
ax.set_ylabel('Height')



