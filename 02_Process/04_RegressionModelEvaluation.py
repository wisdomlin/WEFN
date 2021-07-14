# import utililies
from pandas import read_csv
from matplotlib import pyplot
import numpy as np
from statistics import median
# import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import KFold
# import metrics
from sklearn.metrics import accuracy_score, hamming_loss
# import models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Define X
Year = ['Year']
XFeatures = [
    'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 
    'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 
    'X21', 'X22', 'X23']
XHeader = Year + XFeatures
XFilepath = "01_Input/07_RegressionTraining_ModelInput.csv"
df = read_csv(XFilepath, header=None, names=XHeader, encoding='utf8')
X = df[XHeader].values

# Define Y
YFeatures = ['Water', 'Energy', 'Food', 'Labor', 'Capital']
YHeader = Year + YFeatures
YFilepath = "03_Output/02_ResourcePressureAssessment_ModelOutput.csv"
df = read_csv(YFilepath, header=None, names=YHeader, encoding='utf8')
Y = df[YHeader].values

# define models
models = []
models.append(('LR', LogisticRegression(solver ='lbfgs', max_iter = 400), []))
models.append(('LDA', LinearDiscriminantAnalysis(), []))
models.append(('KNN', KNeighborsClassifier(), []))
models.append(('CART', DecisionTreeClassifier(), []))
models.append(('NB', GaussianNB(), []))
models.append(('SVM', SVC(gamma='auto'), []))

