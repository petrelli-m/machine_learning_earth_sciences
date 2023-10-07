from sklearn import preprocessing
from sklearn.model_selection import train_test_split

le = preprocessing.LabelEncoder()
le.fit(my_data['PGD Type'])
y = le.transform(my_data['PGD Type'])

X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X, y, test_size=0.20) 

X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.25)  