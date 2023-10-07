import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_data = pd.read_hdf('ml_data.h5', 'train')
test_data = pd.read_hdf('ml_data.h5', 'leaderboard_test_features')

show_axes = [1,5,9,13,17]
fig = plt.figure(figsize=(9, 15))

for i, my_feature in enumerate(train_data.columns[0:20], start=1):
    ax = fig.add_subplot(5,4,i)
    min_val = np.nanpercentile(train_data[my_feature],1)
    max_val = np.nanpercentile(train_data[my_feature],99)
    my_bins = np.linspace(min_val,max_val,30)
    ax.hist(train_data[my_feature], bins=my_bins, density = True, 
            histtype='step', color='#0A3A54')
    ax.hist(test_data[my_feature], bins=my_bins, density = True, 
            histtype='step', color='#C82127', linestyle='--')
    ax.set_xlabel(my_feature)
    ymin, ymax = ax.get_ylim()
    if ymax >=10:
        ax.set_yticks(np.round(np.linspace(ymin, ymax, 4), 0))
    elif ((ymax<10)&(ymax>1)):
          ax.set_yticks(np.round(np.linspace(ymin, ymax, 4), 1))
    else:
        ax.set_yticks(np.round(np.linspace(ymin, ymax, 4), 2))

    if i in show_axes:
        ax.set_ylabel('Probability Density') 

plt.tight_layout()
fig.align_ylabels()
