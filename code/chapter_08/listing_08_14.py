import numpy as np
import pandas as pd

A = np.load('penalty_matrix.npy')
def score(y_true, y_pred):
    S = 0.0
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    for i in range(0, y_true.shape[0]):
        S -= A[y_true[i], y_pred[i]]
    return S/y_true.shape[0]

target = np.full(1000, 5) # Limestone
predicted = np.full(1000, 5)  # Limestone
print("Case 1: " + str(score(target, predicted)))

predicted = np.full(1000, 6)  # Chalk
print("Case 2: " + str(score(target, predicted)))

predicted = np.full(1000, 7) # Halite
print("Case 3: " + str(score(target, predicted)))

hidden_test_target = pd.read_hdf('ml_data.h5', 
                                 'hidden_test_target').values
predicted = np.random.randint(0, high=12, 
                              size=1000) # Random predictions
print("Case 4: " + str(score(target, predicted)))

''' Output:
    
Case 1: 0.0
Case 2: -1.375
Case 3: -4.0
Case 4: -3.04625

'''