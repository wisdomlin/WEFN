# make predictions
from pandas import read_csv

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

# Load dataset
names = ['F1', 'F2', 'F3', 'F4', 'Label']
filepath = "01_RawData/01_DomainKnowledge.csv"
dataset = read_csv(filepath, names=names)

# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1, stratify=y)

# # Make predictions on validation dataset
# model = SVC(gamma='auto')
# model.fit(X_train, y_train)
# predictions = model.predict(X_test)

# # Evaluate predictions
# print(accuracy_score(y_test, predictions))
# print(confusion_matrix(y_test, predictions))
# print(classification_report(y_test, predictions))

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# train and test each model in turn
results = []
names = []
for name, model in models:
	model.fit(X_train, y_train)
	predictions = model.predict(X_test)
	results.append(predictions)
	names.append(name)
	print('model: ' + name)
	print(accuracy_score(y_test, predictions))
	print(confusion_matrix(y_test, predictions))
	print(classification_report(y_test, predictions))


