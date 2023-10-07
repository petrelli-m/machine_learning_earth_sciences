import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pandas as pd

leaderboard_test_res= pd.read_hdf('ml_data.h5', 'leaderboard_test_res')
hidden_test_res = pd.read_hdf('ml_data.h5', 'hidden_test_res')

leaderboard_test_target = pd.read_hdf('ml_data.h5', 'leaderboard_test_features_target').values
hidden_test_target = pd.read_hdf('ml_data.h5', 'hidden_test_target').values

leaderboard_accuracy_scores = []
hidden_accuracy_scores = []

for (leaderboard_column, leaderboard_data), (hidden_column, hidden_data) in zip(leaderboard_test_res.iteritems(), hidden_test_res.iteritems()):
    
    leaderboard_accuracy_scores.append(np.around(accuracy_score(leaderboard_data, leaderboard_test_target),2))
    hidden_accuracy_scores.append(np.around(accuracy_score(hidden_data, hidden_test_target),2))


# plot the resultson the test dataset
plt, ax1 = plt.subplots()
labels = leaderboard_test_res.columns
x = np.arange(len(labels))
width = 0.35  
rects1 = ax1.bar(x - width/2, leaderboard_accuracy_scores, width, label='Leaderboard test data set', color='#C82127')
rects2 = ax1.bar(x + width/2, hidden_accuracy_scores, width, label='Hidden test est data set', color='#0A3A54')
ax1.set_ylabel('Accuracy scores')
#ax1.set_xlabel('Model ranking')
ax1.set_ylim(0,1.1)
ax1.set_xticks(x, labels)
ax1.legend()
ax1.bar_label(rects1, padding=3)
ax1.bar_label(rects2, padding=3)


