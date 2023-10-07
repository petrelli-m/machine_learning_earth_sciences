import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

my_array_2d = np.dstack([bands_dict['B2'], 
                         bands_dict['B3'], 
                         bands_dict['B4'],
                         bands_dict['B8']])

my_array_1d =my_array_2d[:,:,:4].reshape(
    (my_array_2d.shape[0] * my_array_2d.shape[1], 
     my_array_2d.shape[2]))

my_array_1d_pandas = pd.DataFrame(my_array_1d, 
                     columns=['B2', 'B3', 'B4', 'B8'])


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7,3))
my_medianprops = dict(color='#C82127', linewidth = 1)
my_boxprops = dict(facecolor='#BFD7EA', edgecolor='#000000')
ax1.boxplot(my_array_1d_pandas, vert=False, whis=(0.5, 99.5), 
            showfliers=False, labels=my_array_1d_pandas.columns,
            patch_artist=True, showcaps=False,
            medianprops=my_medianprops, boxprops=my_boxprops)
ax1.set_xlim(-0.1,0.5)
ax1.set_xlabel('Surface reflectance Value')
ax1.set_ylabel('Band Name')
ax1.grid()
ax1.set_facecolor((0.94, 0.94, 0.94))

colors=['#BFD7EA','#0F7F8B','#C82127','#F15C61']
for band, color in zip(my_array_1d_pandas.columns, colors):
    ax2.hist(my_array_1d_pandas[band], density=True, 
             bins='doane', range=(0,0.5), histtype='step', 
             linewidth=1, fill=True, color=color, alpha=0.6,  
             label=band)
    ax2.hist(my_array_1d_pandas[band], density=True, 
             bins='doane', range=(0,0.5), histtype='step', 
             linewidth=0.5, fill=False, color='k')
ax2.legend(title='Band Name')
ax2.set_xlabel('Surface Reflectance Value')
ax2.set_ylabel('Probability Density')
ax2.xaxis.grid()
ax2.set_facecolor((0.94, 0.94, 0.94))
plt.tight_layout()
plt.savefig('descr_stat_sat.pdf')