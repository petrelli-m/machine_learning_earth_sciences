import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

for my_key in ['Orthopyroxene', 'Liquid_Orthopyroxene']:
    
    fig = plt.figure(figsize=(8,8),constrained_layout=True)
    subfigs = fig.subfigures(nrows=2, ncols=1)
    for j, (trans, my_title) in enumerate(zip(['', '_lrpwt'],
         [my_key, my_key+' log-ratio pairwise transformation'])):
        my_scores = pd.read_hdf('ml_data.h5', 
                                my_key + trans + '_scores')
        
        RMSE_ML_valid_median_T = np.median(
            my_scores['root_mean_squared_error'])
        R2_valid_median_T = np.median(my_scores['r2_score'])
        
        subfigs[j].suptitle(my_title.replace('_', '-'))
        
        # left panel
        ax = subfigs[j].add_subplot(1, 2,1)
        bins = np.arange(30, 70, 2)
        ax.hist(my_scores['root_mean_squared_error'], bins=bins, 
                density = True, color = '#BFD7EA', 
                edgecolor = 'k', 
                label='Hist. distribution')
        ax.axvline(RMSE_ML_valid_median_T, 
                   color='#C82127', 
                   label='Median: {:.0f} °C'.format(
                       RMSE_ML_valid_median_T))
        ax.set_xlabel('Root Mean Squared Error')
        ax.set_ylabel('Prob. Density')
        ax.legend()
        
        # right panel
        ax = subfigs[j].add_subplot(1, 2, 2)
        bins = np.arange(0.875, 1, 0.005)
        ax.hist(my_scores['r2_score'], bins = bins, 
                density = True, color = '#BFD7EA',
                edgecolor='k', 
                label='Hist. distribution')
        ax.axvline(R2_valid_median_T, color='#C82127', 
                   label='Median: {:.2f}'.format(
                       R2_valid_median_T))
        ax.set_xlabel(r'r$^2$ score')
        ax.set_ylabel('Prob. Density')
        ax.legend()