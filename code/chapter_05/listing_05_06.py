#AgglomerativeClustering
model_AC = AgglomerativeClustering(linkage='ward',  
                                affinity='euclidean',
                                n_clusters=6)
my_AC = model_AC.fit(my_dataset_ilr_scaled)

fig, ax = plt.subplots()
label_to_color = [my_colors[i] for i in my_AC.labels_]
ax.scatter(my_PCA[:,0], my_PCA[:,1], 
           c=label_to_color, alpha=0.6, 
           edgecolors='k')
ax.set_title('Hierarchical Clustering')
ax.set_xlabel('PC_1')
ax.set_ylabel('PC_2')
my_dataset['cluster_HC'] = my_AC.labels_

#KMeans
from sklearn.cluster import KMeans
myKM = KMeans(n_clusters=6).fit(my_dataset_ilr_scaled)

fig, ax = plt.subplots()
label_to_color = [my_colors[i] for i in myKM.labels_]
ax.scatter(my_PCA[:,0], my_PCA[:,1], 
           c=label_to_color, alpha=0.6, 
           edgecolors='k')
ax.set_title('KMeans')
ax.set_xlabel('PC_1')
ax.set_ylabel('PC_2')
my_dataset['cluster_KM'] = myKM.labels_