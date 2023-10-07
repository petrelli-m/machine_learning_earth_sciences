import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

def monte_carlo_simulation(X, y, indexes, n, key_res):

    r2 = []
    RMSE = []

    for i in range(n): 
        my_res = {}
        X_train, X_valid,  y_train, y_valid, \
            indexes_train, indexes_valid = train_test_split(
                X, y.ravel(), indexes, test_size=0.2)
        
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_valid = scaler.transform(X_valid)
        
        regressor = ExtraTreesRegressor(n_estimators=450, 
                                        max_features=1).fit(
                                          X_train, y_train)
        my_prediction = regressor.predict(X_valid)

        my_res = {'indexes_valid': indexes_valid,
                  'prediction': my_prediction}
        
        my_res_pd = pd.DataFrame.from_dict(my_res)
        r2.append(r2_score(y_valid, my_prediction))
        RMSE.append(np.sqrt(mean_squared_error(y_valid, 
                                             my_prediction)))
        my_res_pd.to_hdf('ml_data.h5', 
                         key= key_res + '_res_' + str(i)) 

    my_scores = {'r2_score': r2,
                 'root_mean_squared_error': RMSE}
    my_scores_pd = pd.DataFrame.from_dict(my_scores)
    my_scores_pd.to_hdf('ml_data.h5', key = key_res + '_scores') 


my_keys = ['Liquid_Orthopyroxene', 'Liquid_Orthopyroxene_lrpwt']

for my_key in my_keys:
    
    # Liquid plus opx calibration
    liquid_opx = pd.read_hdf('ml_data.h5', my_key)
    print(liquid_opx.columns)
    X_liquid_opx = liquid_opx.values
    my_labels = pd.read_hdf('ml_data.h5', 'labels')
    my_y = my_labels['T(C)'].values
    my_indexes = my_labels['Index'].values
    monte_carlo_simulation(X = X_liquid_opx, y = my_y, 
                           indexes = my_indexes, 
                           n =1000, key_res = my_key)
    
    # opx only calibration
    opx = liquid_opx.loc[:,
                ~liquid_opx.columns.str.startswith('Liquid')]
    X_opx = opx.values
    my_key = my_key.replace("Liquid_", "")
    monte_carlo_simulation(X = X_opx, 
                           y = my_y, indexes = my_indexes, 
                           n =1000, key_res = my_key)