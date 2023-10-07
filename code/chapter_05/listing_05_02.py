from skbio.stats.composition import ilr
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns

elms_for_clustering  = {'cpx':  ['SiO2', 'TiO2', 
            'Al2O3', 'FeO', 'MgO', 'CaO', 'Na2O']}

my_dataset = my_dataset[elms_for_clustering['cpx']]

my_dataset = my_dataset[~((
    my_dataset < my_dataset.quantile(0.001)) | 
    (my_dataset > my_dataset.quantile(0.999))).any(axis=1)]

my_dataset_ilr = ilr(my_dataset)

transformer = RobustScaler(
    quantile_range=(25.0, 75.0)).fit(my_dataset_ilr)

my_dataset_ilr_scaled = transformer.transform(my_dataset_ilr)

fig = plt.figure(figsize=(8,8))

for i in range(0,6):
    ax1 = fig.add_subplot(3, 2, i+1)
    sns.kdeplot(my_dataset_ilr_scaled[:, i],fill=True, 
                color='k', facecolor='#c7ddf4', ax = ax1)
    ax1.set_xlabel('scaled ilr_' + str(i+1))
fig.align_ylabels()
fig.tight_layout()


