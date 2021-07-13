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
# import multiclass
from sklearn.multiclass import OneVsRestClassifier

# common variables
country = ['Country']
features = ['F1', 'F2', 'F3', 'F4']
labels = ['L1', 'L2', 'L3', 'L4']

# load dataset X_train
names = country + features + labels
filepath = "01_Input/02_ClassificationTraining_ModelInput.csv"
df_train = read_csv(filepath, header=None, skiprows=1, names=names, encoding='utf8')

# load dataset X_pred
names = country + features
filepath = "01_Input/04_ClassificationPrediction_ModelInput.csv"
df_pred = read_csv(filepath, header=None, skiprows=1, names=names, encoding='utf8')

# Prepare X
X_train = df_train[features].values
X_pred = df_pred[features].values

# define models
models = []
models.append(('LR', LogisticRegression(solver ='lbfgs', max_iter = 400), []))
models.append(('LDA', LinearDiscriminantAnalysis(), []))
models.append(('KNN', KNeighborsClassifier(), []))
models.append(('CART', DecisionTreeClassifier(), []))
models.append(('NB', GaussianNB(), []))
models.append(('SVM', SVC(gamma='auto'), []))

# majority vote
pred_sumup = {}
# shape 
s = df_pred.values.shape[0]
# print('s:', s)
pred_sumup['L1'] = np.zeros(s)
pred_sumup['L2'] = np.zeros(s)
pred_sumup['L3'] = np.zeros(s)
pred_sumup['L4'] = np.zeros(s)
eclf_trues = []  

# 2. For Models
for name, model, score in models:
    print('model: ' + name)
    # define OvR strategy
    ovr = OneVsRestClassifier(model)
    # multilabel classification
    y_trues = []
    y_preds = []
    # 3. For Labels
    for label in labels:
        # 
        y_train = df_train[label].values        
        # fit model    
        ovr.fit(X_train, y_train)    
        # make predictions
        y_pred = ovr.predict(X_pred)
        print('y_pred is {}'.format(y_pred))
        y_preds.append(y_pred)    
        # pred sum up for eclf
        pred_sumup[label] = np.add(pred_sumup[label], y_pred)
            
# Compute Majority Vote
for key in pred_sumup:
    arr = pred_sumup[key]
    for idx, item in enumerate(arr):            
        # print('pred_sumup: ', item)
        if item <= 3: 
            arr[idx] = 0
        else:
            arr[idx] = 1

# show result as pred_sumup
print('Majority Vote Prediction Result:', pred_sumup)
