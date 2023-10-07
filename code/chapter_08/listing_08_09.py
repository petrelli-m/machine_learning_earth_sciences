import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import joblib as jb
from sklearn.preprocessing import StandardScaler

X = pd.read_hdf('ml_data.h5', 'train').values
y = pd.read_hdf('ml_data.h5', 'train_target').values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

param_grid = {
    'criterion': ['entropy', 'gini'],
    'min_samples_split': [2, 5, 8, 10],
    'max_features': ['sqrt', 'log2', None],
    'class_weight': ['balanced', None]
    }

classifier = ExtraTreesClassifier(n_estimators=250,
                                  n_jobs=-1)

CV_rfc = GridSearchCV(estimator=classifier, param_grid=param_grid, cv= 3, verbose=10)
CV_rfc.fit(X_train, y_train)

jb.dump(CV_rfc, 'ETC_grid_search_results_rev_2.pkl')

