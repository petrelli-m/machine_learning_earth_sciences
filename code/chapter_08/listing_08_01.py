import pandas as pd
import matplotlib.pyplot as plt

data_sets = ['train.csv', 'hidden_test.csv', 'leaderboard_test_features.csv']
labels = ['Train data', 'Hidden test data', 'Leaderboard test data']
colors = ['#BFD7EA','#0A3A54','#C82127']

fig, ax = plt.subplots()

for my_data_set, my_color, my_label in zip(data_sets, colors, labels):
    
    my_data = pd.read_csv(my_data_set, sep=';')
    my_Weels = my_data.drop_duplicates(subset=['WELL'])
    my_Weels = my_Weels[['X_LOC',	'Y_LOC']].dropna() / 100000

    ax.scatter(my_Weels['X_LOC'], my_Weels['Y_LOC'], 
               label=my_label, s=80, color=my_color, 
               edgecolor='k', alpha=0.8)

ax.set_xlabel('X_LOC')
ax.set_ylabel('Y_LOC')
ax.set_xlim(4,6)
ax.set_ylim(63,70)
ax.legend(ncol=3)
plt.tight_layout()