from joblib import load
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CV_rfc = load('ETC_grid_search_results_rev_2.pkl')

my_results = pd.DataFrame.from_dict(CV_rfc.cv_results_)
my_results = my_results.sort_values(by=['rank_test_score'])

# Plot the results of the GridSearch
fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax1.plot(my_results['rank_test_score'], my_results['mean_test_score'], marker='o',
         markeredgecolor='#0A3A54', markerfacecolor='#C82127', color='#0A3A54',
         label='Grid Search Results')
ax1.set_xticks(np.arange(1,50,4))
ax1.invert_xaxis()
ax1.set_xlabel('Model ranking')
ax1.set_ylabel('Accuracy scores')
ax1.legend()

# Selecting the best three performing models
my_results = my_results[my_results['mean_test_score']>0.956]

# Load and scaling
X = pd.read_hdf('ml_data.h5', 'train').values
y = pd.read_hdf('ml_data.h5', 'train_target').values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

leaderboard_test_features = pd.read_hdf('ml_data.h5', 'leaderboard_test_features').values
hidden_test = pd.read_hdf('ml_data.h5', 'hidden_test').values

leaderboard_test_features_scaled = scaler.transform(leaderboard_test_features)
hidden_test_scaled = scaler.transform(hidden_test)

# Apply the three best performing model on the test dataset and on the unknowns
leaderboard_test_res = {}
hidden_test_res  = {} 
test_score  = [] 
rank_model  = [] 
for index, row in my_results.iterrows():
    classifier = ExtraTreesClassifier(n_estimators=250, n_jobs=8, random_state=64, **row['params'])
    classifier.fit(X_train, y_train)
    my_score = classifier.score(X_test,y_test)
    test_score.append(my_score)
    rank_model.append(row['rank_test_score'])
    
    my_leaderboard_test_res = classifier.predict(leaderboard_test_features_scaled)
    my_hidden_test_res = classifier.predict(hidden_test_scaled)
    leaderboard_test_res['model_ranked_' + str(row['rank_test_score'])] = my_leaderboard_test_res
    hidden_test_res['model_ranked_' + str(row['rank_test_score'])] = my_hidden_test_res
 
leaderboard_test_res_pd = pd.DataFrame.from_dict(leaderboard_test_res)
hidden_test_res_pd = pd.DataFrame.from_dict(hidden_test_res)
leaderboard_test_res_pd.to_hdf('ml_data.h5', key= 'leaderboard_test_res')
hidden_test_res_pd.to_hdf('ml_data.h5', key= 'hidden_test_res')

# plot the resultson the test dataset
ax2 = fig.add_subplot(2,1,2)
labels = my_results['rank_test_score']
validation_res = np.around(my_results['mean_test_score'], 2)
test_res = np.around(np.array(test_score),2)
x = np.arange(len(labels))
width = 0.35  
rects1 = ax2.bar(x - width/2, validation_res, width, label='Validation data set', color='#C82127')
rects2 = ax2.bar(x + width/2, test_res, width, label='Test data set', color='#0A3A54')
ax2.set_ylabel('Accuracy scores')
ax2.set_xlabel('Model ranking')
ax2.set_ylim(0,1.7)
ax2.set_xticks(x, labels)
ax2.legend()
ax2.bar_label(rects1, padding=3)
ax2.bar_label(rects2, padding=3)
fig.align_ylabels()
fig.tight_layout()





