import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#######################################
logs_path='C:\\Users\\Aris\\Desktop\\Epilepsy\\logs\\graphing\\Patient_'
patient=7
Headers=['TrainLoss', 'ValidLoss', 'ValidTP', 'ValidFP', 'ValidFN', 'ValidF1_Score', 'TestAccuracy', 'TestTP', 'TestFP', 'TestFN', 'TestF1_Score']
data1 = pd.read_csv(logs_path+str(patient)+'\\TwentyPercentTest\\Test_3(WeightedCross)\\log_log10_priors.csv', names=Headers, header=None)
data2 = pd.read_csv(logs_path+str(patient)+'\\Adam\\log_log10_priors.csv', names=Headers, header=None)
data3 = pd.read_csv(logs_path+str(patient)+'\\WithoutLog\\log_log10_priors.csv', names=Headers, header=None)
data4 = pd.read_csv(logs_path+str(patient)+'\\WithoutPriors\\log_log10_priors.csv', names=Headers, header=None)
data5 = pd.read_csv(logs_path+str(patient)+'\\WithoutDropout\\log_log10_priors.csv', names=Headers, header=None)
#######################################

#######################################
plot1=np.array(data1[['TrainLoss','ValidLoss', 'ValidF1_Score', 'TestF1_Score']].values)
plot2=np.array(data2[['TrainLoss','ValidLoss', 'ValidF1_Score', 'TestF1_Score']].values)
plot3=np.array(data3[['TrainLoss','ValidLoss', 'ValidF1_Score', 'TestF1_Score']].values)
plot4=np.array(data4[['TrainLoss','ValidLoss', 'ValidF1_Score', 'TestF1_Score']].values)
plot5=np.array(data5[['TrainLoss','ValidLoss', 'ValidF1_Score', 'TestF1_Score']].values)

epochs1 = np.linspace(1, np.shape(plot1)[0], np.shape(plot1)[0])
epochs2 = np.linspace(1, np.shape(plot2)[0], np.shape(plot2)[0])
epochs3 = np.linspace(1, np.shape(plot3)[0], np.shape(plot3)[0])
epochs4 = np.linspace(1, np.shape(plot4)[0], np.shape(plot4)[0])
epochs5 = np.linspace(1, np.shape(plot5)[0], np.shape(plot5)[0])
#######################################

minTLoss=sys.float_info.max
for val in plot1[:,0]:
    if(val<=minTLoss):
        minTLoss=val
minTLoss=np.ones(np.shape(epochs1))*minTLoss

minVLoss=sys.float_info.max
for val in plot1[:,1]:
    if(val<=minVLoss):
        minVLoss=val
minVLoss=np.ones(np.shape(epochs1))*minVLoss

maxF1V=sys.float_info.min
for val in plot1[:,2]:
    if(val>=maxF1V):
        maxF1V=val
maxF1V=np.ones(np.shape(epochs1))*maxF1V

maxF1T=sys.float_info.min
for val in plot1[:,3]:
    if(val>=maxF1T):
        maxF1T=val
maxF1T=np.ones(np.shape(epochs1))*maxF1T

#######################################

plt.plot(epochs1,plot1[:,0],linewidth=2)
plt.plot(epochs2,plot2[:,0],linewidth=2)
plt.plot(epochs1,minTLoss)
plt.title('Learning Process')
plt.xlabel('epochs1')
plt.ylabel('TrainLoss')
plt.grid()
plt.legend(['Momentum Opt','Adam Opt'])
plt.show()

plt.plot(epochs1,plot1[:,1],linewidth=2)
plt.plot(epochs2,plot2[:,1],linewidth=2)
plt.plot(epochs1,minVLoss)
plt.title('Learning Process')
plt.xlabel('epochs')
plt.ylabel('ValidLoss')
plt.grid()
plt.legend(['Momentum Opt','Adam Opt'])
plt.show()

plt.plot(epochs1,plot1[:,2],linewidth=2)
plt.plot(epochs2,plot2[:,2],linewidth=2)
plt.plot(epochs1,maxF1V)
plt.title('Learning Process')
plt.xlabel('epochs')
plt.ylabel('ValidF1_Score')
plt.grid()
plt.legend(['Momentum Opt','Adam Opt'])
plt.show()

plt.plot(epochs1,plot1[:,3],linewidth=2)
plt.plot(epochs2,plot2[:,3],linewidth=2)
plt.plot(epochs1,maxF1T)
plt.title('Learning Process')
plt.xlabel('epochs')
plt.ylabel('TestF1_Score')
plt.grid()
plt.legend(['Momentum Opt','Adam Opt'])
plt.show()

#######################################

plt.plot(epochs1,plot1[:,1],linewidth=2)
plt.plot(epochs1,plot1[:,2],linewidth=2)
plt.plot(epochs1,plot1[:,3],linewidth=2)
plt.title('Learning Process')
plt.xlabel('epochs')
plt.ylabel('Learning')
plt.grid()
plt.legend(['Valid Loss','Valid F1-Score','Test F1-Score'])
plt.show()

plt.plot(epochs1,plot1[:,1],linewidth=2)
plt.plot(epochs1,plot1[:,2],linewidth=2)
plt.plot(epochs1,maxF1T)
plt.title('Learning Process')
plt.xlabel('epochs')
plt.ylabel('Learning')
plt.grid()
plt.legend(['Valid Loss','Valid F1-Score','Max Test F1-Score'])
plt.show()

#######################################

Early_Algo = 0.5*plot1[:,1] + 0.5*(1-plot1[:,2])
stop=epochs1
finalF1=plot1[:,3]

plt.plot(epochs1,Early_Algo,linewidth=2)
plt.plot(epochs1,plot1[:,3],linewidth=2)
plt.plot(epochs1,maxF1T)
plt.plot(stop[np.argmin(Early_Algo)],Early_Algo[np.argmin(Early_Algo)],'o',linewidth=2)
plt.title('Learning Process')
plt.xlabel('epochs')
plt.ylabel('Learning')
plt.grid()
plt.legend(['Early Stopping Algo', 'Test F1-Score', 'Max Test F1-Score','stopping'])
plt.show()

print(maxF1T[0])
print(finalF1[np.argmin(Early_Algo)])
