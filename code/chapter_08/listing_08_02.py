import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

lithology_keys = {30000: 'Sandstone',
                 65030: 'Sandstone/Shale',
                 65000: 'Shale',
                 80000: 'Marl',
                 74000: 'Dolomite',
                 70000: 'Limestone',
                 70032: 'Chalk',
                 88000: 'Halite',
                 86000: 'Anhydrite',
                 99000: 'Tuff',
                 90000: 'Coal',
                 93000: 'Basement'}

train_data = pd.read_csv('train.csv', sep=';')

class_abundance = np.vectorize(lithology_keys.get)(
    train_data['FORCE_2020_LITHOFACIES_LITHOLOGY'].values)
unique, counts = np.unique(class_abundance, return_counts=True)

my_colors = ['#0F7F8B'] * len(unique)
my_colors[np.argmax(counts)] = '#C82127'
my_colors[np.argmin(counts)] = '#0A3A54'

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(7,14))

ax2.barh(unique,counts, color=my_colors)
ax2.set_xscale("log")
ax2.set_xlim(1e1,1e6)
ax2.set_xlabel('Number of Occurrences')
ax2.set_title('Class Inspection')

Feature_presence = train_data.isna().sum()/train_data.shape[0]*100

Feature_presence =Feature_presence.drop(
                labels=['FORCE_2020_LITHOFACIES_LITHOLOGY', 
                       'FORCE_2020_LITHOFACIES_CONFIDENCE', 'WELL'])

Feature_presence.sort_values().plot.barh(color='#0F7F8B',ax=ax1)
ax1.axvline(40, color='#C82127', linestyle='--')
ax1.set_xlabel('Percentage of Missing Values')
ax1.set_title('Feature Inspection')

plt.tight_layout()