import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

my_dataset = pd.read_hdf('ml_data.h5', 'Liquid_Orthopyroxene')

Elements  = {
  'Liquid': ['SiO2', 'TiO2', 'Al2O3', 'FeOtot', 'MgO', 
             'MnO', 'CaO', 'Na2O', 'K2O', 'H2O'],
  'Orthopyroxene':  ['SiO2', 'TiO2', 'Al2O3', 'FeOtot', 
             'MgO', 'MnO', 'CaO', 'Na2O', 'K2O', 'Cr2O3']}

fig = plt.figure(figsize=(7,9))
x_labels_melt = [r'SiO$_2$', r'TiO$_2$', r'Al$_2$O$_3$', 
                 r'FeO$_t$', r'MnO', r'MgO', r'CaO', 
                 r'Na$_2O$', r'K$_2$O', r'H$_2$O']
for i, col in enumerate(Elements['Liquid']):
    ax1 = fig.add_subplot(5, 2, i+1)
    sns.kdeplot(my_dataset['Liquid_' + col], fill=True, 
                color='k', facecolor='#BFD7EA', ax = ax1)
    ax1.set_xlabel(x_labels_melt[i] + ' [wt. %] the melt')
    if i in [0,2,4,6,8]:
        ax1.set_ylabel('Prob. Density')
    else:
        ax1.set(ylabel=None)
        
fig.align_ylabels()
fig.tight_layout()

fig1 = plt.figure(figsize=(7,9))
x_labels_cpx = [r'SiO$_2$', r'TiO$_2$', r'Al$_2$O$_3$', 
                r'FeO$_t$', r'MnO', r'MgO', r'CaO', 
                r'Na$_2O$', r'K$_2$O', r'Cr$_2$O$_3$']
for i, col in enumerate(Elements['Orthopyroxene']):
    ax2 = fig1.add_subplot(5, 2, i+1)
    sns.kdeplot(my_dataset['Orthopyroxene_' + col], fill=True, 
                color='k', facecolor='#BFD7EA', ax = ax2)
    ax2.set_xlabel(x_labels_cpx[i] + ' [wt. %] in opx')
    if i in [0,2,4,6,8]:
        ax2.set_ylabel('Prob. Density')
    else:
        ax2.set(ylabel=None)
        
fig1.align_ylabels()
fig1.tight_layout()
