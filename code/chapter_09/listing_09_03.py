import os
import pandas as pd
import numpy as np

Elements  = {
  'Liquid': ['SiO2', 'TiO2', 'Al2O3', 'FeOtot', 'MgO', 
             'MnO', 'CaO', 'Na2O', 'K2O'],
  'Orthopyroxene': ['SiO2', 'TiO2', 'Al2O3', 'FeOtot', 
            'MgO', 'MnO', 'CaO', 'Na2O', 'Cr2O3']}

def calculate_cations_on_oxygen_basis(
        myData0, myphase, myElements, n_oxygens):
    
    Weights = {'SiO2': [60.0843,1.0,2.0], 
               'TiO2':[79.8788,1.0,2.0], 
               'Al2O3': [101.961,2.0,3.0], 
               'FeOtot':[71.8464,1.0,1.0], 
               'MgO':[40.3044,1.0,1.0], 
               'MnO':[70.9375,1.0,1.0], 
               'CaO':[56.0774,1.0,1.0], 
               'Na2O':[61.9789,2.0,1.0],
               'K2O':[94.196,2.0,1.0],
               'Cr2O3':[151.9982,2.0,3.0],
               'P2O5':[141.937,2.0,5.0], 
               'H2O':[18.01388,2.0,1.0]}
    
    myData = myData0.copy()
    # Cation mole proportions
    for el in myElements:
        myData[el + '_cat_mol_prop'] = myData[myphase + 
                   '_' + el] * Weights[el][1] / Weights[el][0]     
    # Oxygen mole proportions
    for el in myElements:
        myData[el + '_oxy_mol_prop'] = myData[myphase + 
                    '_' + el] * Weights[el][2] / Weights[el][0] 
    # Oxigen mole proportions totals
    totals = np.zeros(len(myData.index))
    for el in myElements:
        totals += myData[el + '_oxy_mol_prop']   
    myData['tot_oxy_prop'] = totals 
    # totcations
    totals = np.zeros(len(myData.index))
    for el in myElements:
        myData[el + '_num_cat'] = n_oxygens * myData[el + 
                    '_cat_mol_prop']  /  myData['tot_oxy_prop']
        totals += myData[el + '_num_cat']  
    return totals

def filter_by_cryst_formula(dataFrame, myphase, myElements):
    
    c_o_Tolerance = {'Orthopyroxene': [4,6,0.025]}

    dataFrame['Tot_cations'] = calculate_cations_on_oxygen_basis(
        myData0 = dataFrame, myphase = myphase, 
        myElements = myElements, 
        n_oxygens = c_o_Tolerance[myphase][1])
  
    dataFrame = dataFrame[
        (dataFrame['Tot_cations'] < c_o_Tolerance[myphase][0] 
                             + c_o_Tolerance[myphase][2]) & 
        (dataFrame['Tot_cations'] > c_o_Tolerance[myphase][0] 
                             - c_o_Tolerance[myphase][2])]
  
    dataFrame = dataFrame.drop(columns=['Tot_cations'])
    return dataFrame

def adjustFeOtot(dataFrame):
    for i in range(len(dataFrame.index)):
        try:
            if pd.to_numeric(dataFrame.Fe2O3[i])>0:
                dataFrame.loc[i,'FeOtot'] = (
                    pd.to_numeric(dataFrame.FeO[i]) + 0.8998 * 
                    pd.to_numeric(dataFrame.Fe2O3[i]))     
            else:
                dataFrame.loc[i, 
                    'FeOtot'] = pd.to_numeric(dataFrame.FeO[i]) 
        except:
            dataFrame.loc[i,'FeOtot'] = 0
    return dataFrame

def adjust_column_names(dataFrame):  
    dataFrame.columns = [c.replace('Wt: ', '') 
                         for c in dataFrame.columns]
    dataFrame.columns = [c.replace(' ', '') 
                         for c in dataFrame.columns]
    return dataFrame

def select_base_features(dataFrame, my_elements): 
    dataFrame = dataFrame[my_elements]
    return dataFrame

def data_imputation(dataFrame):
    dataFrame = dataFrame.fillna(0)
    return dataFrame

def pwlr(dataFrame, my_phases):
    
    for my_pahase in my_phases:
        my_indexes  = []
        column_list = Elements[my_pahase]
        
        for col in column_list:
            col = my_pahase + '_' + col
            my_indexes.append(dataFrame.columns.get_loc(col))
            my_min = dataFrame[col][dataFrame[col] > 0].min()
            dataFrame.loc[dataFrame[col] == 0, 
                col] = dataFrame[col].apply(
                lambda x: np.random.uniform(
                    np.nextafter(0.0, 1.0),my_min))
        
        for ix in range(len(column_list)):
            for jx in range(ix+1, len(column_list)):
                col_name = 'log_' + dataFrame.columns[
                    my_indexes[jx]] + '_' + dataFrame.columns[
                        my_indexes[ix]]
                dataFrame.loc[:,col_name] =  np.log(
                  dataFrame[dataFrame.columns[my_indexes[jx]]]/ \
                  dataFrame[dataFrame.columns[my_indexes[ix]]])
    return dataFrame

def data_pre_processing(phase_1, phase_2, out_file): 
  
    try:
        os.remove(out_file)
    except OSError:
        pass
    
    starting = pd.read_excel('LEPR_download.xls', 
                             sheet_name='Experiment')
    starting= adjust_column_names(starting)
    starting.name = ''
    starting = starting[['Index', 'T(C)','P(GPa)']]
    starting.to_hdf(out_file, key='starting_material')  
    
    phases = [phase_1, phase_2]
    
    for ix, my_phase in enumerate(phases):
        my_dataset =  pd.read_excel('LEPR_download.xls', 
                                    sheet_name = my_phase) 
        
        my_dataset = (my_dataset.
                        pipe(adjust_column_names).
                        pipe(adjustFeOtot).
                        pipe(select_base_features, 
                             my_elements= Elements[my_phase]).
                        pipe(data_imputation))
                    
        my_dataset = my_dataset.add_prefix(my_phase + '_')
        my_dataset.to_hdf(out_file, key=my_phase)  
    
    my_phase_1 = pd.read_hdf(out_file, phase_1)
    my_phase_2 = pd.read_hdf(out_file, phase_2)
    
    my_dataset = pd.concat([starting, 
                            my_phase_1, 
                            my_phase_2], axis=1)
    
    my_dataset = my_dataset[(my_dataset['Liquid_SiO2'] > 35)&
                            (my_dataset['Liquid_SiO2'] < 80)]
    
    my_dataset = my_dataset[(
        my_dataset['Orthopyroxene_SiO2'] > 0)]
    
    my_dataset = my_dataset[(my_dataset['P(GPa)'] <= 2)]

    my_dataset = my_dataset[(my_dataset['T(C)'] >= 650)&
                            (my_dataset['T(C)'] <= 1800)] 
   
    my_dataset = filter_by_cryst_formula(dataFrame = my_dataset, 
                                myphase = phase_2, 
                                myElements = Elements[phase_2])
  
    my_dataset = my_dataset.sample(frac=1, 
                         random_state=50).reset_index(drop=True)

    my_labels = my_dataset[['Index', 'T(C)', 'P(GPa)']]
    my_dataset = my_dataset.drop(columns=['T(C)','P(GPa)'])
    
    my_labels.to_hdf(out_file, key='labels') 
    my_dataset.to_hdf(out_file, key= phase_1 + '_' + phase_2) 
    
    my_dataset = pwlr(my_dataset, 
                               my_phases= [phase_1, phase_2])
    my_dataset.to_hdf(out_file, 
                      key= phase_1 + '_' + phase_2 + '_lrpwt') 
    
data_pre_processing(phase_1='Liquid' , 
                    phase_2='Orthopyroxene', 
                    out_file='ml_data.h5')