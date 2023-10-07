import os
import pandas as pd
import numpy as np

def replace_inf(dataFrame):
    to_be_replaced = [np.inf,-np.inf]
    for replace_me in to_be_replaced:
        dataFrame = dataFrame.replace(replace_me, np.nan)
    return dataFrame

def log_transform(dataFrame):
    log_features = ['RDEP','RMED','PEF','GR']
    for my_feature in log_features:
        dataFrame.loc[dataFrame[my_feature] < 0, my_feature] = dataFrame[dataFrame[my_feature] > 0].RDEP.min()
        dataFrame['log_'+ my_feature] = np.log10(dataFrame[my_feature])
    return dataFrame

def calculate_delta(dataFrame):
    delta_features = ['CALI', 'log_RMED', 'log_RDEP', 'RHOB', 
                      'DTC', 'DRHO', 'log_GR' , 'NPHI', 
                      'log_PEF', 'SP']
    wells = dataFrame['WELL'].unique()
    for my_feature in delta_features:
        values = []
        for well in wells:
            my_val = dataFrame[dataFrame['WELL'] == well][my_feature].values
            my_val_ = np.array([my_val[0]] +
                               list(my_val[:-1]))
            delta_my_val = my_val-my_val_
            values = values + list(delta_my_val)
        dataFrame['Delta_' + my_feature] = values
    return dataFrame

def feature_selection(dataFrame):
    features = ['CALI', 'Delta_CALI',  'log_RMED', 
                'Delta_log_RMED', 'log_RDEP', 
                'Delta_log_RDEP', 'RHOB', 'Delta_RHOB',
                'SP', 'Delta_SP', 'DTC', 'Delta_DTC',
                'DRHO', 'Delta_DRHO', 'log_GR', 'Delta_log_GR',
                'NPHI', 'Delta_NPHI', 'log_PEF', 'Delta_log_PEF']        
    dataFrame = dataFrame[features]
    return dataFrame


