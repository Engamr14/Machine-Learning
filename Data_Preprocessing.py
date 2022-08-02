import numpy as np
import scipy
import scipy.linalg
import math
import matplotlib
import matplotlib.pyplot as plt


def mcol (lst):
    return lst.reshape((lst.shape[0],1))

def unique(list1): 
    unique_list = []
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)       
    return sorted(unique_list)

class Data_Preprocessing:
    def __init__(self):
        pass
    
    def load_dataset(self,File):
        Data = []
        Labels = []
        with open(File) as f:
            for line in f:
                try:
                        attrs = line.split(',')[0:-1]
                        attrs = mcol(np.array([float(i) for i in attrs]))
                        label = line.split(',')[-1]
                        Data.append(attrs)
                        Labels.append(label)
                except:
                    pass    
        Data = np.hstack(Data)
        Labels = np.array(Labels, dtype=np.int32)
        return Data, Labels
    
    def split_dataset (self,Data, Labels, small_ratio, permutation_seed=0):
        nTrain = int(Data.shape[1] * (1 - small_ratio))
        np.random.seed(permutation_seed)
        idx = np.random.permutation(Data.shape[1])
        idxTrain = idx[0:nTrain]
        idxTest = idx[nTrain:]
        D_TR = Data[:, idxTrain]
        D_TE = Data[:, idxTest]
        L_TR = Labels[idxTrain]
        L_TE = Labels[idxTest]
        return D_TR, L_TR, D_TE, L_TE
    
    def Dimensionality_Reduction_PCA (self, DataMat, m):
        mu = DataMat.mean(1)
        mu = mcol(mu)
    
        DataCentered = DataMat - mu
        Cov = np.dot(DataCentered,DataCentered.T)
        Cov = Cov / DataCentered.shape[1]
    
        s, U = np.linalg.eigh(Cov)
    
        P = U[:, ::-1][: , 0:m]
    
        DataProjected = np.dot(P.T,DataMat)
        return DataProjected
  
    
    def Dimensionality_Reduction_LDA (self, DataMat, ClassMat, m):
        mu =DataMat.mean(1)
        mu= mcol(mu)
        D=[]
        clsMeans=[]
        n=[]
        SB = []
         
        clss = unique(ClassMat)
        for i in clss:
            D.append(DataMat[:,ClassMat == i])
            clsMeans.append(mcol(D[i].mean(1)))
            n.append(D[i].shape[1])
            SB.append(np.dot((clsMeans[i]-mu),(clsMeans[i]-mu).T))
    
        tot_SB = np.zeros((SB[0].shape[0],SB[0].shape[1]))
        
        for i in clss:
            SB[i] = SB[i]*n[i]
            tot_SB += SB[i]    
    
        tot_SB = tot_SB/DataMat.shape[1]
    
        DC=[]
        SW = []
        
        for i in clss:
            DC.append(D[i]-clsMeans[i])
            SW.append(np.dot(DC[i],DC[i].T))
            
        tot_SW= sum(SW)
        tot_SW = tot_SW/DataMat.shape[1]
        
        s,U = scipy.linalg.eigh(tot_SB,tot_SW)
    
        W = U[:, ::-1][:,0:m]
    
        DataProjected = np.dot(W.T,DataMat)
        return DataProjected
    
    def plot_hist(self, Data, Labels, attributes):
    
        D0 = Data[:, Labels==0]
        D1 = Data[:, Labels==1]
    
        hFea = dict(zip(range(len(attributes)), attributes))
    
        for feature_n in range(Data.shape[0]):
            plt.figure()
            plt.xlabel(hFea[feature_n])
            plt.hist(D0[feature_n, :], bins = 10, density = True, alpha = 0.4, label = 'Male')
            plt.hist(D1[feature_n, :], bins = 10, density = True, alpha = 0.4, label = 'Female')
            
            plt.legend()
            plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
            plt.savefig('../hist_%d.pdf' % feature_n)
        plt.show()
        
        
    def plot_bar(self, x_axis, y_axis, title):
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.bar(x_axis, y_axis)
        ax.set_title(title)
        plt.show()
        
    def plot_normal_distribution(self, Data, title):
        Mean = np.mean(Data, axis = 1)
        Var = np.var(Data, axis = 1)
        max_var = np.max(Var)
        mean_mean = np.mean(Mean)
        x = np.linspace(mean_mean - math.sqrt(max_var), mean_mean + math.sqrt(max_var), 100)
        for mean, var in zip(Mean, Var):
            plt.plot(x, scipy.stats.norm.pdf(x, mean , np.sqrt(var)))
        plt.title(title)
    
    
    def plot_normal_distribution_binary(self, mean0, mean1, var0, var1):
        for i in range(len(mean0)):
            max_var = np.max([var0[i], var1[i]])
            max_mean = np.max([mean0[i], mean1[i]])
            min_mean = np.min([mean0[i], mean1[i]])
            x = np.linspace(min_mean - np.sqrt(max_var), max_mean + np.sqrt(max_var), 100)
            fig, ax = plt.subplots()
            ax.plot(x, scipy.stats.norm.pdf(x, mean0[i] , np.sqrt(var0[i])), title = 'Male')
            ax.plot(x, scipy.stats.norm.pdf(x, mean1[i] , np.sqrt(var1[i])), title = 'Female')
            ax.set_title("attr_" + str(i + 1))
        
        
    
    
    def plot_scatter(self, Data, Labels, attributes):
        D0 = Data[:, Labels==0]
        D1 = Data[:, Labels==1]
    
        hFea = dict(zip(range(len(attributes)), attributes))
    
        for dIdx1 in range(Data.shape[0]):
            for dIdx2 in range(Data.shape[0]):
                if dIdx1 == dIdx2:
                    continue
                plt.figure()
                plt.xlabel(hFea[dIdx1])
                plt.ylabel(hFea[dIdx2])
                plt.scatter(D0[dIdx1, :], D0[dIdx2, :], label = 'Male')
                plt.scatter(D1[dIdx1, :], D1[dIdx2, :], label = 'Female')
            
                plt.legend()
                plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
                plt.savefig('../scatter_%d_%d.pdf' % (dIdx1, dIdx2))
            plt.show()
            
            
            
