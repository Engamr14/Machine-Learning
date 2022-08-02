import numpy as np
import sklearn
from Data_Preprocessing import Data_Preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

data_preprocessing = Data_Preprocessing()
X_train, Y_train = data_preprocessing.load_dataset('Train.txt')
X_test, Y_test = data_preprocessing.load_dataset('Test.txt')

X_train = X_train.T
X_test = X_test.T


logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, Y_train)
accuracy = logistic_regression.score(X_test,Y_test) * 100
print("Logistic Regression accuracy", round(accuracy, 1))


support_vector_machine = SVC()
support_vector_machine.fit(X_train, Y_train)
accuracy = support_vector_machine.score(X_test,Y_test) * 100
print("Support Vector Machine accuracy", round(accuracy, 1))


gaussianNB = GaussianNB()
gaussianNB.fit(X_train, Y_train)
accuracy = gaussianNB.score(X_test,Y_test) * 100
print("Gaussian Naive Bayes accuracy", round(accuracy, 1))