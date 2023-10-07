import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

A = np.load('penalty_matrix.npy')

my_labels = ['Sandstone','Sandstone/Shale','Shale','Marl', 'Dolomite',
             'Limestone','Chalk','Halite','Anhydrite','Tuff','Coal','Basement']

fig, ax = plt.subplots(figsize=(15, 12))
ax.imshow(A)
ax = sns.heatmap(A, annot=True, xticklabels = my_labels, yticklabels = my_labels)
fig.tight_layout()