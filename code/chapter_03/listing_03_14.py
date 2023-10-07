from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

le = preprocessing.LabelEncoder()
le.fit(my_data['PGD Type'])
y = le.transform(my_data['PGD Type'])

parameters = {'kernel':('linear', 'rbf'), 'C':[0.1, 1, 10]}
my_model = svm.SVC()

my_grid_search = GridSearchCV(my_model, parameters,
                        cv = 4, scoring='accuracy')

my_grid_search.fit(scaled_X, y)

