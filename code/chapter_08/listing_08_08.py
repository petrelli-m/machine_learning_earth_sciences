import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder   
from sklearn.impute import SimpleImputer

def replace_inf(dataFrame):
    to_be_replaced = [np.inf,-np.inf]
    for replace_me in to_be_replaced:
        dataFrame = dataFrame.replace(replace_me, np.nan)
    return dataFrame

def log_transform(dataFrame):
    log_features = ['RDEP','RMED','PEF','GR']
    for my_feature in log_features:
        dataFrame.loc[dataFrame[my_feature] < 0, my_feature] = dataFrame[
            dataFrame[my_feature] > 0].RDEP.min()
        dataFrame['log_'+ my_feature] = np.log10(dataFrame[my_feature])
    return dataFrame

def calculate_delta(dataFrame):
    delta_features = ['CALI', 'log_RMED', 'log_RDEP', 'RHOB', 
                      'DTC', 'DRHO', 'log_GR' , 'NPHI', 
                      'log_PEF', 'SP', 'BS']
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

def categorical_encoder(dataFrame, my_encoder, cols):
    dataFrame[cols] =  my_encoder.transform(dataFrame[cols])
    return dataFrame

def feature_selection(dataFrame):
    features = ['CALI', 'Delta_CALI',  'log_RMED', 'Delta_log_RMED',
                'log_RDEP','Delta_log_RDEP', 'RHOB', 'Delta_RHOB',
                'SP', 'Delta_SP', 'DTC', 'Delta_DTC', 'DRHO', 'Delta_DRHO', 
                'log_GR', 'Delta_log_GR', 'NPHI', 'Delta_NPHI', 
                'log_PEF', 'Delta_log_PEF', 'BS', 'Delta_BS', 
                'FORMATION', 'X_LOC','Y_LOC', 'DEPTH_MD']
    dataFrame = dataFrame[features]
    return dataFrame

def pre_processing_pipeline(input_files, out_file): 
    
    try:
        os.remove(out_file)
    except OSError:
        pass
    
    for ix, my_file in enumerate(input_files):
        my_dataset = pd.read_csv(my_file, sep=';')
        
        try:
            my_dataset['FORCE_2020_LITHOFACIES_LITHOLOGY'].to_hdf(
                out_file, key=my_file[:-4] + '_target') 
        except:
            my_target = pd.read_csv('leaderboard_test_target.csv', sep=';')
            my_target['FORCE_2020_LITHOFACIES_LITHOLOGY'].to_hdf(
                out_file, key=my_file[:-4] + '_target') 
                    
        if ix==0:    
            # Fitting the categorical encoders
            my_encoder = OrdinalEncoder()
            my_encoder.set_params(handle_unknown='use_encoded_value', 
                                  unknown_value=-1, 
                                  encoded_missing_value=-1).fit(
                                      my_dataset[['FORMATION']])

        my_dataset = (my_dataset.
                        pipe(replace_inf).
                        pipe(log_transform).
                        pipe(calculate_delta).
                        pipe(categorical_encoder, 
                             my_encoder=my_encoder, cols=['FORMATION']).
                        pipe(feature_selection))
        my_dataset.to_hdf(out_file, key=my_file[:-4])  
        
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputer = imputer.fit(my_dataset[my_dataset.columns])
        my_dataset[my_dataset.columns] = imputer.transform(
            my_dataset[my_dataset.columns])
        my_dataset.to_hdf(out_file, key= my_file[:-4]) 
             
my_files = ['train.csv', 'leaderboard_test_features.csv', 'hidden_test.csv'] 

pre_processing_pipeline(input_files=my_files, out_file='ml_data.h5')


