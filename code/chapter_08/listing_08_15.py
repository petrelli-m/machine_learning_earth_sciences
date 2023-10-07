import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

A = np.load('penalty_matrix.npy')
def score(y_true, y_pred):
    S = 0.0
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    for i in range(0, y_true.shape[0]):
        S -= A[y_true[i], y_pred[i]]
    return S/y_true.shape[0]

lithology_numbers = {30000: 0, 65030: 1, 65000: 2, 80000: 3, 74000: 4, 70000: 5,
                     70032: 6, 88000: 7, 86000: 8, 99000: 9, 90000: 10, 93000: 11}

# Load test data
leaderboard_test_res = pd.read_hdf('ml_data.h5', 'leaderboard_test_res')
hidden_test_res = pd.read_hdf('ml_data.h5', 'hidden_test_res')

leaderboard_test_target = pd.read_hdf('ml_data.h5', 'leaderboard_test_features_target').values
leaderboard_test_target =  np.vectorize(lithology_numbers.get)(leaderboard_test_target) 
hidden_test_target = pd.read_hdf('ml_data.h5', 'hidden_test_target').values
hidden_test_target =  np.vectorize(lithology_numbers.get)(hidden_test_target) 

leaderboard_accuracy_scores = []
hidden_accuracy_scores = []
for (leaderboard_column, leaderboard_data), (hidden_column, hidden_data) in zip(leaderboard_test_res.iteritems(), hidden_test_res.iteritems()):
    
    leaderboard_data =  np.vectorize(lithology_numbers.get)(leaderboard_data) 
    leaderboard_accuracy_scores.append(np.around(score(leaderboard_data, leaderboard_test_target),4))  
    hidden_data =  np.vectorize(lithology_numbers.get)(hidden_data) 
    hidden_accuracy_scores.append(np.around(score(hidden_data, hidden_test_target),4))

# plot the results
plt, ax1 = plt.subplots()
labels = leaderboard_test_res.columns
x = np.arange(len(labels))
width = 0.35  
rects1 = ax1.bar(x - width/2, leaderboard_accuracy_scores, width, label='Leaderboard test data set', color='#C82127')
rects2 = ax1.bar(x + width/2, hidden_accuracy_scores, width, label='Hidden test est data set', color='#0A3A54')
ax1.set_ylabel('Accuracy scores')
ax1.set_ylim(0,-0.7)
ax1.set_xticks(x, labels)
ax1.legend()
ax1.bar_label(rects1, padding=-12)
ax1.bar_label(rects2, padding=-12)

