# import utililies
from pandas import read_csv
# from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
from statistics import median
# import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import KFold
# import metrics
from sklearn.metrics import accuracy_score, hamming_loss
from sklearn.metrics import r2_score, mean_squared_error
# import models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor

# Define X
Year = ['Year']
XFeatures = [
    'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 
    'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 
    'X21', 'X22', 'X23']
XHeader = Year + XFeatures
XFilepath = "01_Input/07_RegressionTraining_ModelInput.csv"
df = read_csv(XFilepath, header=None, skiprows=1, names=XFeatures, encoding='utf8')
X = df[XFeatures].values

# Define Y
YFeatures = ['Water', 'Energy', 'Food', 'Labor', 'Capital']
YHeader = Year + YFeatures
YFilepath = "03_Output/02_ResourcePressureAssessment_ModelOutput.csv"
df = read_csv(YFilepath, header=None, skiprows=1, names=YFeatures, encoding='utf8')
Y_Water = df['Water'].values
Y_Energy = df['Energy'].values
Y_Food = df['Food'].values

# Y Setting
Y = Y_Water

# train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, 
                                     train_size = 0.8, 
                                     test_size = 0.2, 
                                     shuffle = False)

# Define models
reg1 = GradientBoostingRegressor(random_state=1)
reg2 = RandomForestRegressor(random_state=1)
reg3 = LinearRegression()
ereg = VotingRegressor([('gb', reg1), ('rf', reg2), ('lr', reg3)])

# TODO: Need to Refresh the model and then do the next training?


# Training 
reg1.fit(x_train, y_train)
reg2.fit(x_train, y_train)
reg3.fit(x_train, y_train)
ereg.fit(x_train, y_train)

# Making Predictions
# xt = X[:2]
# x_test = X
pred1 = reg1.predict(x_test)
pred2 = reg2.predict(x_test)
pred3 = reg3.predict(x_test)
pred4 = ereg.predict(x_test)

# # Plot the results
# plt.figure()
# plt.plot(pred1, 'gd', label='GradientBoostingRegressor')
# plt.plot(pred2, 'b^', label='RandomForestRegressor')
# plt.plot(pred3, 'ys', label='LinearRegression')
# plt.plot(pred4, 'r*', ms=10, label='VotingRegressor')

# plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
# plt.ylabel('predicted')
# plt.xlabel('training samples')
# plt.legend(loc="best")
# plt.title('Regressor predictions and their average')
# plt.show()

# R2
# r2_1 = r2_score(y_test, pred1)
# r2_2 = r2_score(y_test, pred2)
# r2_3 = r2_score(y_test, pred3)
# r2_4 = r2_score(y_test, pred4)
# print(r2_1)
# print(r2_2)
# print(r2_3)
# print(r2_4)

# MSE for test
print('MSE for test')
MSE_1 = mean_squared_error(y_test, pred1)
MSE_2 = mean_squared_error(y_test, pred2)
MSE_3 = mean_squared_error(y_test, pred3)
MSE_4 = mean_squared_error(y_test, pred4)
print(MSE_1)
print(MSE_2)
print(MSE_3)
print(MSE_4)

# MSE for train
print('MSE for train')
pred1 = reg1.predict(x_train)
pred2 = reg2.predict(x_train)
pred3 = reg3.predict(x_train)
pred4 = ereg.predict(x_train)
MSE_1 = mean_squared_error(y_train, pred1)
MSE_2 = mean_squared_error(y_train, pred2)
MSE_3 = mean_squared_error(y_train, pred3)
MSE_4 = mean_squared_error(y_train, pred4)
print(MSE_1)
print(MSE_2)
print(MSE_3)
print(MSE_4)

# print y_test & Pred
print('print y_test & Pred')
print(["{0:0.6f}".format(i) for i in y_test])

print(["{0:0.6f}".format(i) for i in pred1])
print(["{0:0.6f}".format(i) for i in pred2])
print(["{0:0.6f}".format(i) for i in pred3])
print(["{0:0.6f}".format(i) for i in pred4])





