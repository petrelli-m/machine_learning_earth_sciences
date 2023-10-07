import numpy as np
import pandas as pd

def calc_cations_on_oxygen_basis(myData0, my_ph, my_el, n_ox):
    Weights = {
        'SiO2': [60.0843,1.0,2.0], 'TiO2':[79.8788,1.0,2.0], 
        'Al2O3': [101.961,2.0,3.0],'FeO':[71.8464,1.0,1.0], 
        'MgO':[40.3044,1.0,1.0], 'MnO':[70.9375,1.0,1.0], 
        'CaO':[56.0774,1.0,1.0], 'Na2O':[61.9789,2.0,1.0],
        'K2O':[94.196,2.0,1.0], 'Cr2O3':[151.9982,2.0,3.0],
        'P2O5':[141.937,2.0,5.0], 'H2O':[18.01388,2.0,1.0]}
    myData = myData0.copy()
    myData = myData.add_prefix(my_ph + '_')
    for el in my_el: # Cation mole proportions
        myData[el + '_cat_mol_prop'] = myData[my_ph + 
                '_' + el] * Weights[el][1] / Weights[el][0]   
    for el in my_el:  # Oxygen mole proportions
        myData[el + '_oxy_mol_prop'] = myData[my_ph + 
                '_' + el] * Weights[el][2] / Weights[el][0]
    totals = np.zeros(len(myData.index)) # Ox mole prop tot
    for el in my_el:
        totals += myData[el + '_oxy_mol_prop']
    myData['tot_oxy_prop'] = totals
    totals = np.zeros(len(myData.index)) # totcations
    for el in my_el:
        myData[el + '_num_cat'] = n_ox * myData[el + 
                    '_cat_mol_prop']  /  myData['tot_oxy_prop']
        totals += myData[el + '_num_cat']
    return totals

my_dataset = pd.read_table('ETN21_cpx_all.txt')
my_dataset = my_dataset[(my_dataset.Total>98) & 
                        (my_dataset.Total<102)]
Elements  = {'cpx':  ['SiO2', 'TiO2', 'Al2O3', 
            'FeO', 'MgO', 'MnO', 'CaO', 'Na2O','Cr2O3']}
Cat_Ox_Tolerance = {'cpx':  [4,6,0.06]}
my_dataset['Tot_cations'] = calc_cations_on_oxygen_basis(
            myData0 = my_dataset, 
            my_ph = 'cpx', 
            my_el = Elements['cpx'], 
            n_ox = Cat_Ox_Tolerance['cpx'][1])

my_dataset = my_dataset[(
    my_dataset['Tot_cations'] < Cat_Ox_Tolerance['cpx'][0] + 
    Cat_Ox_Tolerance['cpx'][2])&(
    my_dataset['Tot_cations'] > Cat_Ox_Tolerance['cpx'][0] - 
    Cat_Ox_Tolerance['cpx'][2])]