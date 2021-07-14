import matplotlib.pyplot as plt
import numpy as np
from pandas.io.parsers import read_csv

# Resource Pressure Assessment Algorithm
class NSA:
  def __init__(self, CRT, W1, W2, E1, E2, F1, F2, L1, L2, C1, C2):
    self.CRT = CRT
    self.W1 = float(W1)
    self.W2 = float(W2)
    self.E1 = float(E1)
    self.E2 = float(E2)
    self.F1 = float(F1)
    self.F2 = float(F2)
    self.L1 = float(L1)
    self.L2 = float(L2)
    self.C1 = float(C1)
    self.C2 = float(C2)

  def run(self):
    if self.CRT == 3:
      Water = ((self.W1-0.40)/0.40+2)*2/5 + ((self.W2-0.29)/0.29+2)*3/5      
      Energy = ((self.E1-0.88)/0.88+2)*3/5 + ((self.E2-0.80)/0.80+2)*2/5      
      Food = ((self.F1-13.523)/13.523+2)*2/4 + ((self.F2-(-0.073))/(-0.073)+2)*2/4      
      Labor = ((self.L1-1.42)/1.42+2)*1/4 + ((self.L2-0.045)/0.045+2)*3/4      
      Capital = ((self.C1-0.096)/0.096+2)*1/4 + ((self.C2-0.223)/0.223+2)*3/4      
      return Water, Energy, Food, Labor, Capital

# Define Input
Year = ['Year']
InputFeatures = ['W1', 'W2', 'E1', 'E2', 'F1', 'F2', 'L1', 'L2', 'C1', 'C2']
InputHeader = Year + InputFeatures
InputFilepath = "01_Input/06_ResourcePressureAssessment_ModelInput.csv"
df = read_csv(InputFilepath, header=None, names=InputHeader, encoding='utf8')
Input = df[InputHeader].values

# Define Output
OutputFeatures = ['Water', 'Energy', 'Food', 'Labor', 'Capital']
OutputHeader = Year + OutputFeatures
OutputFilepath = "03_Output/02_ResourcePressureAssessment_ModelOutput.csv"
OutputSize_X = np.size(OutputHeader)
OutputSize_Y = np.size(Input, 0)
Output = [[0]*OutputSize_X for _ in range(OutputSize_Y)]

# Define Process 
Output[0] = OutputHeader
Country_type = 3
for i in range(1, OutputSize_Y):
  Nsa = NSA(Country_type, 
            Input[i][1], Input[i][2], Input[i][3], Input[i][4], Input[i][5], 
            Input[i][6], Input[i][7], Input[i][8], Input[i][9], Input[i][10])
  W, E, F, L, C = Nsa.run()
  Output[i][0] = Input[i][0]
  Output[i][1] = round(W,3)
  Output[i][2] = round(E,3)
  Output[i][3] = round(F,3)
  Output[i][4] = round(L,3)
  Output[i][5] = round(C,3)
  
# Save Output 
np.savetxt(OutputFilepath, Output, fmt="%s", delimiter=',')
  