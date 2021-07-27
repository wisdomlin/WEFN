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

# Define Subsystems
Subsystems = []
Subsystems.append(('Water', Y_Water))
Subsystems.append(('Energy', Y_Energy))
Subsystems.append(('Food', Y_Food))

# Define Models
M_GB = GradientBoostingRegressor(random_state=1)
M_RF = RandomForestRegressor(random_state=1)
M_LR = LinearRegression()
M_Vote = VotingRegressor([('gb', M_GB), ('rf', M_RF), ('lr', M_LR)])

Models = []
Models.append(('GradientBoostingRegressor', M_GB))
Models.append(('RandomForestRegressor', M_RF))
Models.append(('LinearRegression', M_LR))
Models.append(('VotingRegressor', M_Vote))

# Residual Containers
Residual_Subsystems = []
Ypred_Subsystems = []

# For each Subsystem: 
for Y_name, Y in Subsystems: 
  # train_test_split
  print('Y_name:', Y_name)
  x_train, x_test, y_train, y_test = train_test_split(X, Y, 
                                    train_size = 0.8, 
                                    test_size = 0.2, 
                                    shuffle = False)
  
  scores = []
  Residual_Models = []
  Ypred_Models = []
# For each Model: 
  for m_name, m in Models: 
    print('m_name:', m_name)
    # Train 
    m.fit(x_train, y_train)
    
    # Test
    y_test_pred = m.predict(x_test)
    
    # Train MSE
    y_train_pred = m.predict(x_train)
    MSE_train = mean_squared_error(y_train, y_train_pred)
    # print('y_train:', y_train)
    print('y_train_pred:', y_train_pred)
    # print('MSE_train:', "{:.2e}".format(MSE_train))
    
    # Test MSE
    MSE_test = mean_squared_error(y_test, y_test_pred)
    # print('y_test:', y_test)
    print('y_test_pred:', y_test_pred)
    # print('MSE_test:', "{:.2e}".format(MSE_test))
        
    # Score Append
    scores.append(MSE_test)
        
    # Residual  
    Residual_Train = []  
    zip_object = zip(y_train, y_train_pred)
    for list1_i, list2_i in zip_object:
      # print('list1_i:', "{:.10e}".format(list1_i))
      # print('list2_i:', "{:.10e}".format(list2_i))
      # print('list1_i - list2_i:', "{:.10e}".format(list1_i - list2_i))
      Residual_Train.append(list1_i - list2_i)

    Residual_Test = []  
    zip_object = zip(y_test, y_test_pred)
    for list1_i, list2_i in zip_object:
      # print('list1_i:', "{:.10e}".format(list1_i))
      # print('list2_i:', "{:.10e}".format(list2_i))
      # print('list1_i - list2_i:', "{:.10e}".format(list1_i - list2_i))
      Residual_Test.append(list1_i - list2_i)

    Residual = Residual_Train + Residual_Test
    Residual_Models.append (Residual)
    # print('Residual:', Residual)
     
    # Ypred 
    # Ypred = y_train_pred + y_test_pred
    Ypred = np.concatenate([y_train_pred, y_test_pred]).tolist()
    Ypred_Models.append(Ypred)
    print('Ypred:', Ypred)
    
    # # print arrays
    # print('y_test: ')
    # print(["{0:0.6f}".format(i) for i in y_test])
    # print('y_test_pred: ')
    # print(["{0:0.6f}".format(i) for i in y_test_pred])
    # print('y_train: ')
    # print(["{0:0.6f}".format(i) for i in y_train])
    # print('y_train_pred: ')
    # print(["{0:0.6f}".format(i) for i in y_train_pred])    

  # Compare Scores
  # print(np.argsort(scores), '\n')
  
  Residual_Subsystems.append(Residual_Models)
  Ypred_Subsystems.append(Ypred_Models)

# print('Residual_Subsystems:', Residual_Subsystems)
print('Ypred_Subsystems:', Ypred_Subsystems)
# TODO: Need to Refresh the model and then do the next training?


# Residual Plot 



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






