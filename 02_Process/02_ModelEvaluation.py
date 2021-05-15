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

# load dataset
country = ['Country']
features = ['F1', 'F2', 'F3', 'F4']
labels = ['L1', 'L2', 'L3', 'L4']
names = country + features + labels
filepath = "01_RawData/02_ClassificationTraining_ModelInput.csv"
df = read_csv(filepath, header=None, skiprows=1, names=names, encoding='utf8')
X = df[features].values

# define models
models = []
models.append(('LR', LogisticRegression(solver ='lbfgs', max_iter = 400), []))
models.append(('LDA', LinearDiscriminantAnalysis(), []))
models.append(('KNN', KNeighborsClassifier(), []))
models.append(('CART', DecisionTreeClassifier(), []))
models.append(('NB', GaussianNB(), []))
models.append(('SVM', SVC(gamma='auto'), []))

# define ensemble classifier
eclf_HS_score = []

# define cross validation
cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=2652124)

# 1. For nFolds
cnt = 1
for train_index, test_index in cv.split(X):
    print('#', cnt)
    cnt += 1
    # print("TRAIN:", train_index)
    # print("TEST:", test_index)
    # 
    pred_sumup = {}
    shape = df[labels].values[test_index].shape[0]
    print('shape:', shape)
    # 
    pred_sumup['L1'] = np.zeros(shape=(4,1))
    pred_sumup['L2'] = np.zeros(shape=(4,1))
    pred_sumup['L3'] = np.zeros(shape=(4,1))
    pred_sumup['L4'] = np.zeros(shape=(4,1))
    eclf_preds = []
    eclf_trues = []  
    # 2. For Models
    for name, model, score in models:
        print('model: ' + name)              
        # define OvR strategy
        ovr = OneVsRestClassifier(model)
        # multilable classification
        y_trues = []
        y_preds = []
        # 3. For Labels
        for label in labels:
            y = df[label].values
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            # fit model    
            ovr.fit(X_train, y_train)    
            # make predictions
            y_pred = ovr.predict(X_test)
            print('y_pred is {}'.format(y_pred))
            # pred sum up for eclf
            pred_sumup[label] = np.add(pred_sumup[label], y_pred)
            y_preds.append(y_pred)
            print('y_true is {}'.format(y_test))
            y_trues.append(y_test)
            # print('Test accuracy is {}'.format(accuracy_score(y_test, y_pred)))
        hamming_score = 1-hamming_loss(y_trues, y_preds)
        score.append(hamming_score)
        print('Hamming score (label-based accuracy) is ', '{0:.4f}'.format(hamming_score))  
        # print('y_trues is {}'.format(y_trues))
        eclf_trues = y_trues
    
    # Compute Majority Vote Score  
    for key in pred_sumup:            
        for arr in pred_sumup[key]:
            print('pred_sumup: ', arr)
            for i in range(len(arr)):
                if arr[i] <= 3: 
                    arr[i] = 0
                else:
                    arr[i] = 1
        eclf_preds.append(arr)             
    eclf_HS = 1-hamming_loss(eclf_trues, eclf_preds)
    eclf_HS_score.append(eclf_HS) 
    print('Hamming score of Majority Vote is ', '{0:.4f}'.format(eclf_HS))  

# clf
model_scores = []
model_names = []
for name, model, score in models:
    model_scores.append(score) 
    model_names.append(name)
    print("model:", name, "\t", 
          "max:", "{0:.3f}".format(max(score)), 
          "min:", "{0:.3f}".format(min(score)), 
          "avg:", "{0:.3f}".format(sum(score)/len(score)), 
          "median:", "{0:.3f}".format(median(score)))

# eclf
model_scores.append(eclf_HS_score) 
model_names.append('eclf')
print("model:", 'eclf', "\t", 
        "max:", "{0:.3f}".format(max(eclf_HS_score)), 
        "min:", "{0:.3f}".format(min(eclf_HS_score)), 
        "avg:", "{0:.3f}".format(sum(eclf_HS_score)/len(eclf_HS_score)), 
        "median:", "{0:.3f}".format(median(eclf_HS_score)))
    
# model evaluation
pyplot.boxplot(model_scores, showmeans=True, labels=model_names)
pyplot.title('Model Evaluation')
pyplot.show()