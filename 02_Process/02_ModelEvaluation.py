from pandas import read_csv
from matplotlib import pyplot
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import KFold

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, hamming_loss

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import LinearSVC, SVC

from sklearn.multiclass import OneVsRestClassifier
from statistics import median

# load dataset
country = ['Country']
features = ['F1', 'F2', 'F3', 'F4']
labels = ['L1', 'L2', 'L3', 'L4']
names = country + features + labels
filepath = "01_RawData/02_ClassificationTraining_ModelInput.csv"
df = read_csv(filepath, header=None, skiprows=1, names=names, encoding='utf8')

# split train/test set
# train, test = train_test_split(df, random_state=42000, test_size=0.20, shuffle=True)

# define models
models = []
models.append(('LR', LogisticRegression(solver ='lbfgs', max_iter = 400), []))
models.append(('LDA', LinearDiscriminantAnalysis(), []))
models.append(('KNN', KNeighborsClassifier(), []))
models.append(('CART', DecisionTreeClassifier(), []))
models.append(('NB', GaussianNB(), []))
models.append(('SVM', SVC(gamma='auto'), []))

# define nFolds
cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=2652124)

X = df[features].values

# 1. For nFolds
for train_index, test_index in cv.split(X):
    # print("TRAIN:", train_index)
    # print("TEST:", test_index)
    # 2. For Models
    for name, model, score in models:
        # print('model: ' + name)
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
            # print('y_pred is {}'.format(y_pred))
            y_preds.append(y_pred)
            # print('y_true is {}'.format(y_test))
            y_trues.append(y_test)
            # print('Test accuracy is {}'.format(accuracy_score(y_test, y_pred)))
        hamming_score = 1-hamming_loss(y_trues, y_preds)
        score.append(hamming_score)
        # print('Hamming score (label-based accuracy) is {}'.format(hamming_score))   
     
     
# # ------------------------- backup
# # train and test each model in turn
# for name, model in models:    
#     print('model: ' + name)
#     # define OvR strategy
#     ovr = OneVsRestClassifier(model)
#     # multilable classification
#     y_trues = []
#     y_preds = []
#     for label in labels:
#         # print('... Processing label: {}'.format(label))
#         X_train = train[features].values	
#         y_train = train[label].values
#         X_test = test[features].values
#         y_test = test[label].values
#         # fit model    
#         ovr.fit(X_train, y_train)    
#         # make predictions
#         y_pred = ovr.predict(X_test)
#         # print('y_pred is {}'.format(y_pred))
#         y_preds.append(y_pred)
#         # print('y_true is {}'.format(y_test))
#         y_trues.append(y_test)
#         # print('Test accuracy is {}'.format(accuracy_score(y_test, y_pred)))

#     # multilabel evaluation metrics
#     # print('Subset accuracy (exact match ratio) is {}'.format(accuracy_score(y_trues, y_preds)))
#     print('Hamming score (label-based accuracy) is {}'.format(1-hamming_loss(y_trues, y_preds)))
# # -------------------------


# evaluate each model in turn

# names = []
# for name, model in models:
#     print('model: ' + name)
#     results = []
#     # perform sampling for each label
#     for label in labels:
#         # print('... Processing label: {}'.format(label))      
#         X = df[features].values
#         y = df[label].values
#         # X_train = train[features].values
#         # y_train = train[label].values
#         # X_test = test[features].values
#         # y_test = test[label].values
#         cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=36851234)
#         cv_results = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
#         print('%s: %f (%f)' % (label, cv_results.mean(), cv_results.std()))   
#         results.append(cv_results.mean()) 
    
#     # numpy.array
#     print('%s: %f' % (name, sum(results) / len(results)))    
#     names.append(name)

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
    
# Model Evaluation
pyplot.boxplot(model_scores, showmeans=True, labels=model_names)
pyplot.title('Model Evaluation')
pyplot.show()