import numpy as np
from Data_Preprocessing import Data_Preprocessing
import scipy.stats as stats
import math
import matplotlib.pyplot as plt

data_preprocessing = Data_Preprocessing()

Data, Labels = data_preprocessing.load_dataset("Train.txt")

attributes = ['attr_1','attr_2', 'attr_3', 'attr_4', 'attr_5', 'attr_6',
              'attr_7', 'attr_8', 'attr_9', 'attr_10', 'attr_11', 'attr_12']


mean0 = np.mean(Data[:, Labels == 0], axis = 1)
mean1 = np.mean(Data[:, Labels == 1], axis = 1)
var0 = np.var(Data[:, Labels == 0], axis = 1)
var1 = np.var(Data[:, Labels == 1], axis = 1)

data_preprocessing.plot_normal_distribution_binary(mean0, mean1, var0, var1)


data_preprocessing.plot_bar(attributes, mean0, 'Male Mean')
data_preprocessing.plot_bar(attributes, mean1, 'Female Mean')

data_preprocessing.plot_bar(attributes, var0, 'Male Variance')
data_preprocessing.plot_bar(attributes, var1, 'Female Variance')

data_preprocessing.plot_hist(Data, Labels, attributes)

data_preprocessing.plot_scatter(Data, Labels, attributes)

data_preprocessing.plot_normal_distribution(Data, 'Normal distribution of data')

reduced_data=data_preprocessing.Dimensionality_Reduction_PCA(Data,8)
reduced_data=data_preprocessing.Dimensionality_Reduction_LDA(reduced_data,Labels,2)

data_preprocessing.plot_scatter(reduced_data, Labels, ['Dim0', 'Dim1'])
data_preprocessing.plot_normal_distribution(reduced_data, 'Normal distribution of reduced-dimension data')

mean0 = np.mean(reduced_data[:, Labels == 0], axis = 1)
mean1 = np.mean(reduced_data[:, Labels == 1], axis = 1)
var0 = np.var(reduced_data[:, Labels == 0], axis = 1)
var1 = np.var(reduced_data[:, Labels == 1], axis = 1)

data_preprocessing.plot_normal_distribution_binary(mean0, mean1, var0, var1)







