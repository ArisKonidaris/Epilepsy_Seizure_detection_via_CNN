import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#######################################
logs_path='C:\\Users\\Username\\Desktop\\Epilepsy\\logs\\graphing\\Patient_'
patient=7
Headers=['TrainLoss', 'ValidLoss', 'ValidTP', 'ValidFP', 'ValidFN', 'ValidF1_Score', 'TestAccuracy', 'TestTP', 'TestFP', 'TestFN', 'TestF1_Score']
data = pd.read_csv(logs_path+str(patient)+'\\TwentyPercentTest\\Test_3(WeightedCross)\\log_log10_priors.csv', names=Headers, header=None)
#######################################

plot1=np.array(data[['TrainLoss','ValidLoss', 'ValidF1_Score', 'TestF1_Score']].values)
epochs = np.linspace(1, np.shape(plot1)[0], np.shape(plot1)[0])

minTLoss=10000
for val in plot1[:,0]:
    if(val<=minTLoss):
        minTLoss=val
minTLoss=np.ones(np.shape(epochs))*minTLoss

minVLoss=10000
for val in plot1[:,1]:
    if(val<=minVLoss):
        minVLoss=val
minVLoss=np.ones(np.shape(epochs))*minVLoss

maxF1V=0
for val in plot1[:,2]:
    if(val>=maxF1V):
        maxF1V=val
maxF1V=np.ones(np.shape(epochs))*maxF1V

maxF1T=0
for val in plot1[:,3]:
    if(val>=maxF1T):
        maxF1T=val
maxF1T=np.ones(np.shape(epochs))*maxF1T

#######################################

plt.plot(epochs,plot1[:,0],linewidth=2)
plt.plot(epochs,minTLoss)
plt.title('Learning Process')
plt.xlabel('Epochs')
plt.ylabel('TrainLoss')
plt.grid()
plt.show()

plt.plot(epochs,plot1[:,1],linewidth=2)
plt.plot(epochs,minVLoss)
plt.title('Learning Process')
plt.xlabel('Epochs')
plt.ylabel('ValidLoss')
plt.grid()
plt.show()

plt.plot(epochs,plot1[:,2],linewidth=2)
plt.plot(epochs,maxF1V)
plt.title('Learning Process')
plt.xlabel('Epochs')
plt.ylabel('ValidF1_Score')
plt.grid()
plt.show()

plt.plot(epochs,plot1[:,3],linewidth=2)
plt.plot(epochs,maxF1T)
plt.title('Learning Process')
plt.xlabel('Epochs')
plt.ylabel('TestF1_Score')
plt.grid()
plt.show()

#plt.plot(epochs[110:],plot1[110:,2],linewidth=2)
#plt.legend(['ValidLoss','ValidF1_Score','TestF1_Score'])

#######################################

plt.plot(epochs,plot1[:,1],linewidth=2)
plt.plot(epochs,plot1[:,2],linewidth=2)
plt.plot(epochs,plot1[:,3],linewidth=2)
plt.title('Learning Process')
plt.xlabel('Epochs')
plt.ylabel('Learning')
plt.grid()
plt.legend(['Valid Loss','Valid F1-Score','Test F1-Score'])
plt.show()

plt.plot(epochs,plot1[:,1],linewidth=2)
plt.plot(epochs,plot1[:,2],linewidth=2)
plt.plot(epochs,maxF1T)
plt.title('Learning Process')
plt.xlabel('Epochs')
plt.ylabel('Learning')
plt.grid()
plt.legend(['Valid Loss','Valid F1-Score','Max Test F1-Score'])
plt.show()

#######################################

Early_Algo = 0.5*plot1[:,1] + 0.5*(1-plot1[:,2])
stop=epochs
finalF1=plot1[:,3]

plt.plot(epochs,Early_Algo,linewidth=2)
plt.plot(epochs,plot1[:,3],linewidth=2)
plt.plot(epochs,maxF1T)
plt.plot(stop[np.argmin(Early_Algo)],Early_Algo[np.argmin(Early_Algo)],'o',linewidth=2)
plt.title('Learning Process')
plt.xlabel('Epochs')
plt.ylabel('Learning')
plt.grid()
plt.legend(['Early Stopping Algo', 'Test F1-Score', 'Max Test F1-Score','stopping'])
plt.show()

print(maxF1T[0])
print(finalF1[np.argmin(Early_Algo)])
