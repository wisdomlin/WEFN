# import utililies
from pandas import read_csv
import matplotlib.pyplot as plt
import numpy as np

# import model_selection
from sklearn.model_selection import train_test_split

# import metrics
from sklearn.metrics import mean_squared_error

# import models
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

# -------------------------------------------
# Part 1: Model Training and Testing
# -------------------------------------------

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
Models = []
M_GB = GradientBoostingRegressor(random_state=1)
Models.append(('GradientBoostingRegressor', M_GB))

M_RF = RandomForestRegressor(random_state=1)
Models.append(('RandomForestRegressor', M_RF))

M_LR = LinearRegression()
Models.append(('LinearRegression', M_LR))

M_Vote = VotingRegressor([('gb', M_GB), ('rf', M_RF), ('lr', M_LR)])
Models.append(('VotingRegressor', M_Vote))

M_KNN = KNeighborsRegressor(n_neighbors=3, weights='distance', radius=1)
Models.append(('KNeighborsRegressor', M_KNN))

M_DT = DecisionTreeRegressor(max_depth=10, random_state=1)
Models.append(('DecisionTreeRegressor', M_DT))

# Subsystems Metadata Containers
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
    
  # Models Metadata Containers  
  Scores_Models = []
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
    # print('y_train_pred:', y_train_pred)
    # print('MSE_train:', "{:.2e}".format(MSE_train))
    
    # Test MSE
    MSE_test = mean_squared_error(y_test, y_test_pred)
    # print('y_test:', y_test)
    # print('y_test_pred:', y_test_pred)
    # print('MSE_test:', "{:.2e}".format(MSE_test))
        
    # Score Append
    Scores_Models.append(MSE_test)
        
    # Residual Computations
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
    Ypred = np.concatenate([y_train_pred, y_test_pred]).tolist()
    Ypred_Models.append(Ypred)
    # print('Ypred:', Ypred)  

  # Compare Scores
  print('Model Ranking:', np.argsort(Scores_Models), '\n')
  
  Residual_Subsystems.append(Residual_Models)
  Ypred_Subsystems.append(Ypred_Models)

# print('Residual_Subsystems:', Residual_Subsystems)
# print('Ypred_Subsystems:', Ypred_Subsystems)

# -------------------------------------------
# Part 2: Residual Plot Drawing
# -------------------------------------------

# Residual Plot
plt.figure()
nrows = len(Models)
ncols = len(Subsystems)
Sn = 1
Mn = 0
Axs = []
Labels = ['Water','Energy','Food']

# For each Subsystem: 
for Sn_Ypred, Sn_Residual in zip(Ypred_Subsystems, Residual_Subsystems): 
  # Plotting Configurations
  if Sn==1:
    color = 'steelblue'
    marker = 'o'
    s = 100
    xmin = 2.1
    xmax = 2.3
  elif Sn==2:
    color = 'orange'
    marker = '*'
    s = 200
    xmin = 2.06
    xmax = 2.12
  elif Sn==3: 
    color = 'limegreen'
    marker = 's'  
    s = 80
    xmin = 1.1
    xmax = 1.42
  
  # For each Model: 
  for Mn_Ypred, Mn_Residual in zip(Sn_Ypred, Sn_Residual):
    # Plotting Configurations
    if Mn==0:
      title = 'Gradient Boosting'
    elif Mn==1:
      title = 'Random Forest'
    elif Mn==2: 
      title = 'Linear Regression' 
    elif Mn==3:
      title = 'Voting Regressor'
    elif Mn==4: 
      title = 'K Nearest Neighbors'
    elif Mn==5:
      title = 'Decision Tree Regressor' 
    
    # Subplot Indexing and Drawing
    PltIdx = Mn*ncols + Sn
    plt.subplot(nrows, ncols, PltIdx)
    ax = plt.scatter(Mn_Ypred, Mn_Residual, c=color, marker=marker, s=s, edgecolor='white')
    if Mn==0:
      Axs.append(ax)
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.gca().set_title(title)
    plt.gca().set_xlim(xmin=xmin, xmax=xmax)
    yabs_max = abs(max(plt.gca().get_ylim(), key=abs))
    plt.gca().set_ylim(ymin=-yabs_max*1.2, ymax=yabs_max*1.2) 
    plt.hlines(y=0, xmin=xmin, xmax=xmax, color='black', lw=2)
    
    # Subplot Indexing 
    Mn += 1
  
  # Subplot Indexing
  Mn = 0
  Sn += 1

# Plotting Configurations
plt.subplots_adjust(top=0.914,
                    bottom=0.054,
                    left=0.05,
                    right=0.987,
                    hspace=1.0,
                    wspace=0.214)
plt.figlegend(handles=Axs, labels=Labels, 
                    loc='upper center', ncol=len(Labels), prop={'size': 14})

# Plot the results
plt.show()
