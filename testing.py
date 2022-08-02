from Support_Vector_Machine import Support_Vector_Machine
from Logistic_Regression import Logistic_Regression
from Multivariate_Gaussian import Multivariate_Gaussian
from Naive_Bayes_Gaussian import Naive_Bayes_Gaussian
from Tied_Covariance_Gaussian import Tied_Covariance_Gaussian
from Data_Preprocessing import Data_Preprocessing
from Testing_Tools import Testing_Tools

##################### Instantiating Tools and Models #################################
data_preprocessing = Data_Preprocessing()
testing_tools = Testing_Tools()
logistic_regression = Logistic_Regression()
support_vector_machine = Support_Vector_Machine()
multivariate_gaussian = Multivariate_Gaussian()
naive_bayes_gaussian = Naive_Bayes_Gaussian()
tied_covariance_gaussian = Tied_Covariance_Gaussian()

########################## Loading Datasets ##########################################
X_train, Y_train = data_preprocessing.load_dataset('Train.txt')
X_test, Y_test = data_preprocessing.load_dataset('Test.txt')

'''
print("\n############## Multivariate Gaussian #################")
evaluation_accuracy = testing_tools.Kfold_Cross_Validation(multivariate_gaussian, X_train, Y_train, K = 7)
testing_accuracy = testing_tools.Test_Model(multivariate_gaussian, X_train, Y_train, X_test, Y_test)

print("\n############## Naive Bayes Gaussian ##################")
evaluation_accuracy = testing_tools.Kfold_Cross_Validation(naive_bayes_gaussian, X_train, Y_train, K = 7)
testing_accuracy = testing_tools.Test_Model(naive_bayes_gaussian, X_train, Y_train, X_test, Y_test)

print("\n############# Tied Covariance Gaussian ###############")
evaluation_accuracy = testing_tools.Kfold_Cross_Validation(tied_covariance_gaussian, X_train, Y_train, K = 7)
testing_accuracy = testing_tools.Test_Model(tied_covariance_gaussian, X_train, Y_train, X_test, Y_test)
'''
print("\n############## Logistic Regression ###################")
evaluation_accuracy = testing_tools.Kfold_Cross_Validation(logistic_regression, X_train, Y_train,K =  7)
testing_accuracy = testing_tools.Test_Model(logistic_regression, X_train, Y_train, X_test, Y_test)

'''
print("\n############## Support Vector Machine #################")
evaluation_accuracy = testing_tools.Kfold_Cross_Validation(support_vector_machine, X_train, Y_train, K = 7)
testing_accuracy = testing_tools.Test_Model(support_vector_machine, X_train, Y_train, X_test, Y_test)


print("_________________After applying PCA and LDA ______________________")

X_train=data_preprocessing.Dimensionality_Reduction_PCA(X_train,8)
X_train=data_preprocessing.Dimensionality_Reduction_LDA(X_train,Y_train,2)

X_test =data_preprocessing.Dimensionality_Reduction_PCA(X_test,8)
X_test =data_preprocessing.Dimensionality_Reduction_LDA(X_test,Y_test,2)

print("\n############## Multivariate Gaussian #################")
evaluation_accuracy = testing_tools.Kfold_Cross_Validation(multivariate_gaussian, X_train, Y_train, K = 7)
testing_accuracy = testing_tools.Test_Model(multivariate_gaussian, X_train, Y_train, X_test, Y_test)


print("\n############## Naive Bayes Gaussian ##################")
evaluation_accuracy = testing_tools.Kfold_Cross_Validation(naive_bayes_gaussian, X_train, Y_train, K = 7)
testing_accuracy = testing_tools.Test_Model(naive_bayes_gaussian, X_train, Y_train, X_test, Y_test)


print("\n############## Tied Covariance Gaussian #################")
evaluation_accuracy = testing_tools.Kfold_Cross_Validation(tied_covariance_gaussian, X_train, Y_train, K = 7)
testing_accuracy = testing_tools.Test_Model(tied_covariance_gaussian, X_train, Y_train, X_test, Y_test)

print("\n############## Logistic Regression ###################")
evaluation_accuracy = testing_tools.Kfold_Cross_Validation(logistic_regression, X_train, Y_train,K =  7)
testing_accuracy = testing_tools.Test_Model(logistic_regression, X_train, Y_train, X_test, Y_test)


print("\n############## Support Vector Machine #################")
evaluation_accuracy = testing_tools.Kfold_Cross_Validation(support_vector_machine, X_train, Y_train, K = 7)
testing_accuracy = testing_tools.Test_Model(support_vector_machine, X_train, Y_train, X_test, Y_test)

'''


