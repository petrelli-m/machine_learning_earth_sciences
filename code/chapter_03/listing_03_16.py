import numpy as np
from sklearn import svm
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt

loo = LeaveOneOut()

my_model = svm.SVC(kernel='linear', C=1, random_state=42)

cv_results = cross_validate(my_model, scaled_X, y, cv=loo,
                            scoring='accuracy')

fig, ax = plt.subplots()
my_x = [0,1]
my_height = [np.count_nonzero(cv_results['test_score'] == 0),
             np.count_nonzero(cv_results['test_score'] == 1)]    
my_bar = ax.bar(x = my_x, height=my_height, width=1, 
                color=['#F15C61', '#BFD7EA'], 
                tick_label=['wrongly classified', 'correcty classified'], 
                edgecolor='k')
ax.set_ylabel('occurrences')
ax.set_title('LOO cross validation n = {}'.format(len(scaled_X)))
ax.bar_label(my_bar)
ax.set_ylim(0,1600)
