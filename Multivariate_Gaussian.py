import numpy as np 
import math

def mcol(lst):
    return lst.reshape((lst.shape[0],1))

def mrow (lst):
    return lst.reshape((1,lst.shape[0]))

def Compute_Full_Covariance (X,mu):

    Cov=np.zeros((X.shape[0],X.shape[0]));
    mu = mcol(mu)
    
    for i in range(X.shape[1]):
        x = X[:,i:i+1]    
        oneElement = np.dot((x-mu),(x-mu).T)
        Cov += oneElement
    
    Cov /= X.shape[1]
    return Cov

def Log_PDF (X,mu,C):
    Y = np.empty([X.shape[0], X.shape[1]])
    sign,logC = np.linalg.slogdet(C)
    Cinv = np.linalg.inv(C)

    for i in range(X.shape[1]):
        x = X[:,i:i+1]
        M = x.shape[0]
        logN = -M/2 * math.log(2*math.pi)
        logN -= 1/2 *logC
        pram = np.dot((x-mu).T,Cinv)
        logN -= 1/2 * np.dot(pram,(x-mu))
        
        Y[0,i] = logN[0][0]

    return Y[0]

class Multivariate_Gaussian:
    def __init__(self):
        pass
        
    def fit(self, DTR, LTR):
        self.muC=[]
        self.cov=[]
        self.D=[]
        self.difClasses = np.unique(LTR)
        for clsno in self.difClasses:
            self.D.append(DTR[:,LTR == clsno])
            self.muC.append(self.D[clsno].mean(1))
            self.cov.append(Compute_Full_Covariance(self.D[clsno],self.muC[clsno]))
            
            
    def predict(self, DTE):
        S=[]
        
        for clsno in self.difClasses:
            Y = Log_PDF(DTE,mcol(self.muC[clsno]),self.cov[clsno])
            Y = np.exp(Y)
            S.append(Y)
            
        S = np.array(S, dtype=np.float32)
        SJoint = S / len(self.difClasses)

        SMarginal=mrow(SJoint.sum(axis = 0))
        SPost = SJoint/SMarginal
        Labels=np.argmax(SPost, axis=0)

        return Labels
    

    
    
    
    
    
    
    
    
