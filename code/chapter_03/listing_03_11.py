import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture as GMM

my_colors = ['#AF41A5','#0A3A54','#0F7F8B','#BFD7EA','#F15C61',
            '#C82127','#ADADAD','#FFFFFF', '#EABD00']

scaler = StandardScaler().fit(X)
scaled_X = scaler.transform(X)

my_model = GMM(n_components = 9, random_state=(42)).fit(scaled_X)

Y = my_model.predict(scaled_X)

fig, ax = plt.subplots()

for my_group in np.unique(Y):
    i = np.where(Y == my_group)
    ax.scatter(scaled_X[i,0], scaled_X[i,1], 
               color=my_colors[my_group], 
               label=my_group + 1 ,  edgecolor='k', alpha=0.8)
    
ax.legend(title='Cluster')

ax.set_xlabel(r'$log_{10}[^{12}C/^{13}C]$')  
ax.set_ylabel(r'$log_{10}[^{14}N/^{15}N]$')  

fig.tight_layout()