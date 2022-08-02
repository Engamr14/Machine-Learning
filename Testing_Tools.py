import numpy as np

class Testing_Tools:
    def __init__(self):
        pass
    
    def Compute_Accuracy(self, Y, Y_predict):
        compare = Y_predict[Y_predict == Y]
        accuracy = compare.shape[0]/Y.shape[0] *100
        return accuracy
    
    def Test_Model(self, model, X_train, Y_train, X_test, Y_test):
        model.fit(X_train, Y_train)
        results = model.predict(X_test)
        accuracy = self.Compute_Accuracy(Y_test, results)
        print("Testing Accuracy = ", round(accuracy, 1)," %")
        return accuracy
    
    def Kfold_Cross_Validation(self, model, X, Y, K):
        n_attributes, n_samples = X.shape
        X = X.T
        n_samples_per_fold = int(n_samples/K)
        starting_index = 0
        ending_index = n_samples_per_fold
        total_accuracy = 0
        for i in range(K):
            # Compute the testing samples
            X_test = X[starting_index : ending_index]
            Y_test = Y[starting_index : ending_index]
            
            # Compute the training samples
            X_train_part1 = X[0 : starting_index]
            X_train_part2 = X[ending_index: -1]
            X_train = np.concatenate((X_train_part1, X_train_part2), axis = 0)
            
            Y_train_part1 = Y[0 : starting_index]
            Y_train_part2 = Y[ending_index: -1]
            Y_train = np.concatenate((Y_train_part1, Y_train_part2), axis = 0)
            
            # Apply to the model and get accuracy
            model.fit(X_train.T, Y_train)

            
            results = model.predict(X_test.T)
            total_accuracy += self.Compute_Accuracy(Y_test, results)
            
            # Updating indexes for next iteration
            starting_index += n_samples_per_fold
            ending_index += n_samples_per_fold
            
        avg_accuracy = total_accuracy/K
        print("Evaluation Accuracy = ", round(avg_accuracy, 1)," %")
        
        return avg_accuracy
    
            
    
    
    
    
    
    
    
    
    
