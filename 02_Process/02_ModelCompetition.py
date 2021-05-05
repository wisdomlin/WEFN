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
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.multiclass import OneVsRestClassifier

# Load dataset
features = ['F1', 'F2', 'F3', 'F4']
labels = ['L1', 'L2', 'L3', 'L4']
# names = ['Country'] + features + labels
filepath = "01_RawData/01_DomainKnowledge.csv"
# df = read_csv(filepath, names=names)
df = read_csv(filepath, header=0)

# array = df.values
# # Features (Column 1~4)
# X = array[:,1:5]	
# # Label (Column 5~8)
# y = array[:,5:9]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

train, test = train_test_split(df, random_state=42, test_size=0.20, shuffle=True)
# X_train = train.values[:,1:5]	
# X_test = test.values[:,5:9]
X_train = train[features]	
X_test = test[features]

# y_train = train.values[:,1:5]	
# y_test = test.values[:,5:9]



# # define model
# model = LogisticRegression()
# # define OvR strategy
# ovr = OneVsRestClassifier(model)
# # fit model
# ovr.fit(X_train, y_train)
# # make predictions
# yhat = ovr.predict(X_test)

ytrues = []
yhats = []
ovr = OneVsRestClassifier(LogisticRegression(solver ='lbfgs', max_iter = 400))
for label in labels:
    print('... Processing {}'.format(label))
    # train the model using X_dtm & y
    # ovr.fit(X_train.values, train[label].values)
    ovr.fit(train[features], train[label])
    # compute the testing accuracy
    yhat = ovr.predict(test[features])
    # yhats.append(yhat)
    # np.concatenate((yhats,yhat), axis=1)
    # print('ytrue is {}'.format(test[label].values))
    ytrues.append(test[label].values)
    # print('yhat is {}'.format(yhat))
    yhats.append(yhat)
    print('Test accuracy is {}'.format(accuracy_score(test[label], yhat)))

print('ytrues is {}'.format(ytrues))
print('yhats is {}'.format(yhats))

# y_true = test[labels]

print('Test accuracy is {}'.format(accuracy_score(ytrues, yhats)))

#-----------------------------------------------------------------------------
# Evaluate predictions
# print(accuracy_score(y_test, yhat))
# print(confusion_matrix(y_test, yhat))
# print(classification_report(y_test, yhat))




# models = []
# models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC(gamma='auto')))

# train and test each model in turn
# results = []
# names = []
# for name, model in models:
# 	model.fit(X_train, y_train)
# 	predictions = model.predict(X_test)
# 	results.append(predictions)
# 	names.append(name)
# 	print('model: ' + name)
# 	print(accuracy_score(y_test, predictions))
# 	print(confusion_matrix(y_test, predictions))
# 	print(classification_report(y_test, predictions))


