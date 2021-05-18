#FAO Method for Taiwan(2006~2016) (v20210108)
"""
import csv
with open(‘2.1_input_Taiwan.csv’, ‘r’) as file:
	reader = csv.reader(file)
	DBS= [row for row in reader]
	#print(DBS)
"""

class NSA:
  def __init__(self, CRT, Year, W1, W2, E1, E2, F1, F2, L1, L2, C1, C2):
    self.CRT = CRT
    self.Year = float(Year)
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

  def NSA_run(self):
    if self.CRT == 3:
      Water=((self.W1-0.40)/0.40+2)*2/5+((self.W2-0.29)/0.29+2)*3/5
      Energy=((self.E1-0.88)/0.88+2)*3/5+((self.E2-0.80)/0.80+2)*2/5
      Food=((self.F1-13523)/13523+2)*2/4+((self.F2-(-0.0730))/(-0.0730)+2)*2/4
      Labor=((self.L1-1.42)/1.42+2)*1/4+((self.L2-0.045)/0.045+2)*3/4
      Capital=((self.C1-0.096)/0.096+2)*1/4+((self.C2-0.223)/0.223+2)*3/4
      return Water, Energy, Food, Labor, Capital


class DRG:
  def __init__(self, Water, Energy, Food, Labor, Capital):
    self.Water = float(Water)
    self.Energy = float(Energy)
    self.Food = float(Food)
    self.Labor = float(Labor)
    self.Capital = float(Capital)
  
  def DRG_run(self):
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    plt.style.use('ggplot')
    values = [self.Water, self.Energy, self.Food, self.Labor, self.Capital]
    feature = ['Water','Energy','Food','Labor','Capital']
    angles=np.linspace(0, 2*np.pi,len(values), endpoint=False)
    values=np.concatenate((values,[values[0]]))
    angles=np.concatenate((angles,[angles[0]]))
    fig=plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    # ax.set_thetagrids(angles * 180/np.pi, feature)
    ax.set_ylim(0,3)
    plt.title(DBS[i][0])
    ax.grid(True)
    plt.show()

"""
-------------------------------------------------------------------------------------------------------
"""
import numpy as np
DBS= [['', 'W1', 'W2', 'E1', 'E2', 'F1', 'F2', 'L1', 'L2', 'C1', 'C2'], 
    ['2006', '0.245617997', '0.255', '0.8999', '0.9808', '1.419349712', '-0.020220635', '1.767807542', '0.054791811', '0.0000847540 ', '0.2461'], 
    ['2007', '0.197849852', '0.262', '0.9028', '0.9813', '1.571257747', '0.001359116', '1.785599372', '0.052749174', '0.0000899452 ', '0.2403'], 
    ['2008', '0.229071634', '0.299', '0.8985', '0.9805', '1.811365883', '-0.045838453', '1.805747095', '0.051427473', '0.0000841168 ', '0.2446'], 
    ['2009', '0.295075547', '0.332', '0.8957', '0.9812', '1.45979251', '0.020258473', '1.681615244', '0.05282615', '0.0000764811 ', '0.1991'], 
    ['2010', '0.273624914', '0.297', '0.8994', '0.9814', '1.832000979', '-0.07530825', '1.843181958', '0.052415896', '0.0000760627 ', '0.2496'], 
    ['2011', '0.293812498', '0.318', '0.895', '0.9803', '2.1091698', '0.096211975', '1.848996671', '0.050611635', '0.0000749702 ', '0.2339'], 
    ['2012', '0.19894951', '0.31', '0.8993', '0.9787', '1.992585679', '0.021231381', '1.84507302', '0.050092081', '0.0000736405 ', '0.2235'], 
    ['2013', '0.243689074', '0.344', '0.8984', '0.9793', '1.987354278', '0.039620329', '1.824661908', '0.049603356', '0.0000686928 ', '0.2218'], 
    ['2014', '0.384600369', '0.326', '0.9009', '0.9808', '2.138186999', '0.022920261', '1.859485457', '0.049462948', '0.0000602092 ', '0.2168'], 
    ['2015', '0.305158518', '0.352', '0.9105', '0.9799', '1.969603501', '-0.12007659', '1.909378239', '0.049562422', '0.0000576556 ', '0.2083'], 
    ['2016', '0.187905287', '0.371', '0.9192', '0.9792', '1.924304421', '0.151947175', '1.882716298', '0.049436407', '0.0000536008 ', '0.209']]
DBS_Y=np.size(DBS,0)
DBS_X=np.size(DBS,1)

Country_type = 3
chart=[[0]*6 for _ in range(DBS_Y)] 
chart[0][0]=" "
chart[0][1]="Water"
chart[0][2]="Energy"
chart[0][3]="Food"
chart[0][4]="Labor"
chart[0][5]="Capita"


for i in range(1,DBS_Y):
  #NSA
  nsa = NSA(Country_type, DBS[i][0],DBS[i][1],DBS[i][2],DBS[i][3],DBS[i][4],DBS[i][5],DBS[i][6],DBS[i][7],DBS[i][8],DBS[i][9],DBS[i][10])
  nsa.NSA_run()
  W, E, F, L, C = nsa.NSA_run()
  chart[i][1]=round(W,3)
  chart[i][2]=round(E,3)
  chart[i][3]=round(F,3)
  chart[i][4]=round(L,3)
  chart[i][5]=round(C,3)
  
  chart[i][0]=DBS[i][0]
  print(chart[i][0])
  print("Water:" + str(chart[i][1]) + "; Energy: " + str(chart[i][2]) + "; Food: " + str(chart[i][3]) + "; Labor: " +str(chart[i][4]) + "; Capita: " + str(chart[i][5]))
  #DRG
  radar_graph = DRG(W, E, F, L, C)
  radar_graph.DRG_run()
  
#   input("Press Enter To Continue...")

for i in range(0,DBS_Y):
  print (chart[i])