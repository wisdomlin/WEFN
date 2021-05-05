from pandas import read_csv
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import LinearSVC, SVC

from sklearn.multiclass import OneVsRestClassifier

# load dataset
country = ['Country']
features = ['F1', 'F2', 'F3', 'F4']
labels = ['L1', 'L2', 'L3', 'L4']
names = country + features + labels
filepath = "01_RawData/01_DomainKnowledge.csv"
df = read_csv(filepath, header=None, skiprows=1, names=names, encoding='utf8')

# split train/test set
train, test = train_test_split(df, random_state=42, test_size=0.20, shuffle=True)

#  from 29 to 59 for backup
# # define model
# model = LogisticRegression(solver ='lbfgs', max_iter = 400)
# # define OvR strategy
# ovr = OneVsRestClassifier(model)

# # multilable classification
# y_trues = []
# y_preds = []
# for label in labels:
#     # print('... Processing label: {}'.format(label))
#     X_train = train[features].values	
#     y_train = train[label].values
#     X_test = test[features].values
#     y_test = test[label].values
#     # fit model    
#     ovr.fit(X_train, y_train)    
#     # make predictions
#     y_pred = ovr.predict(X_test)
#     # print('ytrue is {}'.format(test[label].values))
#     y_preds.append(y_pred)
#     # print('yhat is {}'.format(yhat))
#     y_trues.append(y_test)
#     # evaluate predictions
#     print('Test accuracy is {}'.format(accuracy_score(y_test, y_pred)))

# # print('y_trues is {}'.format(y_trues))
# # print('y_preds is {}'.format(y_preds))
# print('multilable classification accuracy_score is {}'.format(accuracy_score(y_trues, y_preds)))
# # print(confusion_matrix(y_trues, y_preds))
# # print(classification_report(y_trues, y_preds))

#-----------------------------------------------------------------------------
models = []
models.append(('LR', LogisticRegression(solver ='lbfgs', max_iter = 400)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# train and test each model in turn
results = []
names = []
for name, model in models:    
    print('model: ' + name)
    names.append(name)
	# model.fit(X_train, y_train)
	# predictions = model.predict(X_test)
    ovr = OneVsRestClassifier(model)
    
    # multilable classification
    y_trues = []
    y_preds = []
    for label in labels:
        # print('... Processing label: {}'.format(label))
        X_train = train[features].values	
        y_train = train[label].values
        X_test = test[features].values
        y_test = test[label].values
        # fit model    
        ovr.fit(X_train, y_train)    
        # make predictions
        y_pred = ovr.predict(X_test)
        # print('ytrue is {}'.format(test[label].values))
        y_preds.append(y_pred)
        # print('yhat is {}'.format(yhat))
        y_trues.append(y_test)
        # evaluate predictions
        print('Test accuracy is {}'.format(accuracy_score(y_test, y_pred)))

    # print('y_trues is {}'.format(y_trues))
    # print('y_preds is {}'.format(y_preds))
    print('multilable classification accuracy_score is {}'.format(accuracy_score(y_trues, y_preds)))	
	# print(accuracy_score(y_test, predictions))
	# print(confusion_matrix(y_test, predictions))
	# print(classification_report(y_test, predictions))




