import numpy as np

class Support_Vector_Machine:
    def __init__(self, learning_rate=0.0001, n_iters=1000, lambda_param=0.001, C = 1):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.C  = C
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.T.shape

        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0
        
        dW = np.zeros(n_features)
        db = 0
        

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X.T):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    dW = 2 * self.lambda_param * self.w
                    db = 0
                else:
                    dW = 2 * self.lambda_param * self.w - self.C * np.dot(x_i, y_[idx])
                    db = self.C * y_[idx]
                    
                self.w -= self.lr * dW
                self.b -= self.lr * db
        

    def predict(self, X):
        approx = np.dot(X.T, self.w) - self.b
        results = np.where(np.ones(approx.shape)- approx > 0, 0, 1)
        return results
    

    