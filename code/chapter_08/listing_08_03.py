import numpy as np

fig = plt.figure(figsize=(8,4))

train_data['log_RDEP'] = np.log10(train_data['RDEP'])

to_be_plotted = ['RDEP', 'log_RDEP']

for index, my_feature in enumerate(to_be_plotted):
    ax = fig.add_subplot(1,2,index+1)
    min_val = np.nanpercentile(train_data[my_feature],1)
    max_val = np.nanpercentile(train_data[my_feature],99)
    my_bins = np.linspace(min_val,max_val,30)
    ax.hist(train_data[my_feature], bins=my_bins, 
            density = True,  color='#BFD7EA',
            edgecolor='k')
    ax.set_ylabel('Probability Density')
    ax.set_xlabel(my_feature)

plt.tight_layout()
    