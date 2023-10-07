from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

my_colors = {0:'#0A3A54', 
          1:'#E08B48',
          2:'#BFBFBF', 
          3:'#BD22C6', 
          4:'#FD787B',
          5:'#67CF62' }
#PCA
model_PCA = PCA()
model_PCA.fit(my_dataset_ilr_scaled)
my_PCA = model_PCA.transform(my_dataset_ilr_scaled)

fig, ax = plt.subplots()

ax.scatter(my_PCA[:,0], my_PCA[:,1], 
           alpha=0.6, 
           edgecolors='k')

ax.set_title('Principal Component Analysys')
ax.set_xlabel('PC_1')
ax.set_ylabel('PC_2')