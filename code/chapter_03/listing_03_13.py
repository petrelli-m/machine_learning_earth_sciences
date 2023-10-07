from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import cross_validate

le = preprocessing.LabelEncoder()
le.fit(my_data['PGD Type'])
y = le.transform(my_data['PGD Type'])

my_model = svm.SVC(kernel='linear', C=1, random_state=42)

cv_results = cross_validate(my_model, scaled_X, y, cv=5,
                            scoring='accuracy')

print(cv_results['test_score'])

'''
Output:
[0.98529412 0.97785978 0.9704797  0.98154982 0.95940959]    
'''
