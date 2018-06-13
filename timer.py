from datetime import datetime, timedelta
import time
from notest import notest
from notraining import notraining
import marketOrder as mO
#What are we working with?
import sys
sys.version
#Import Libraries
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import random
import orderChecker
from pandas import DataFrame
from pandas import concat

import tensorflow as tf
import shutil
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.layers as tflayers
from tensorflow.contrib.learn.python.learn import learn_runner
import tensorflow.contrib.metrics as metrics
import tensorflow.contrib.rnn as rnn
#Main File for doing shit
#import panda
import talib
import algo.get
import algo.getpast
import common.config
import common.args
import getFullpast
from trainOpt import trainOpt
from tradeModel import tradeModel
from notraining import notraining
from getSpread import getSpread
#import datetime
#while 1:




def train(pairs,predInset,x):

    b=0
    trainTime=[]
    for p in pairs:
        inputData.append(getFullpast.getFullpast(p,"M15",11))
        trainTime=datetime.now()
        
    for p in pairs:    
        for predIn in predInset:   
            spread = getSpread(p)
            notest(p,predIn,inputData[b], spread, x[1], x[2], x[3],None, x[5])
            dt = datetime.now() + timedelta(hours=1)
            dt = dt.replace(minute=00, second=00,  microsecond=4)
        b=b+1
    
    return inputData, trainTime, dt


#pairIn,predIn,A,spread, hidden, layers_stacked_count, beta,stoploss,m


#train models

pairs=['EUR_USD', "EUR_GBP", "GBP_USD", "EUR_JPY", "AUD_JPY", "USD_JPY", "AUD_USD"]
#pairs=['EUR_USD']
inputDataRaw=[]
inputData=[]
predInset=[2]
#predInset=[1]



x=[]
t=0
x = [2, #pred
     18, #nodes
     1, #stacks
     0.013, #beta
     0.15, #threshold
     2.5,#training multiplier
     2.7,     #stop multi
     1.7] #profitmulti
     
seqlength=100

#Train neural networks
inputData, trainTime, dt = train(pairs,predInset,x)
dt = datetime.now()    
dt = datetime.now() +  timedelta(minutes=(15-dt.minute%15))      
dt = dt.replace(second=00,  microsecond=00)
while datetime.now() < dt:
    time.sleep(1)


while 1:
    B=datetime.now()
    t= (B.hour - trainTime.hour)*4+(np.floor(B.minute/15) - np.floor(trainTime.minute/15)) #time difference between now and training
    t=int(t)
    if t<100:
        inputDataRaw=[]       
        #get latest candles      
        a=0
        for p in pairs:
            inputDataRaw.append(getFullpast.getFullpast(p,"M15",1))  
           
            totResults=[]
            for predIn in predInset:
                temp=[]    
                temp.append(inputData[a])        
                if(t>0):
                    temp=(np.array(temp))
                    temp=np.squeeze(temp)
                    A1=inputDataRaw[a][-(t):,:].reshape(t,-1)
                    temp=np.concatenate((temp,A1),axis=0)
                else:
                    temp=(np.array(temp))
                    temp=np.squeeze(temp)
                
            #pad out     
                A=len(inputData[0])+seqlength-len(temp)
                temp=np.concatenate((temp,np.zeros((A,6))))
        #        inputData[a]=temp
                results=(notraining(temp,p,predIn, x[1], x[2], x[3]))
                results=results[-seqlength+int(t),:]
                totResults.append(results)

                np.savetxt("C:\\Users\\Mikw\\Google Drive\\forex\\"+p+"_"+str(predIn)+".csv", results, delimiter=",",fmt="%s", newline='\n')
#                np.savetxt("C:\\Users\\Mikw\\Google Drive\\forex\\time.csv", timeLog, delimiter=",",fmt="%s", newline='\n')
            a=a+1
            totResults=np.array(totResults)
            new=np.sum(totResults, axis=0)/len(totResults)
            if new[0]>x[4] and new[1]<new[0]:
                mO.placeOrder(p,"buy",x[6],x[7])
            elif new[1]>x[4] and new[0]<new[1]:   
                mO.placeOrder(p,"sell",x[6],x[7])
            
        dt = datetime.now()    
        dt = datetime.now() +  timedelta(minutes=(15-dt.minute%15))
        
        dt = dt.replace(second=00,  microsecond=00)
        #        t=t+1
        while datetime.now() < dt:
            orderChecker.orderChecker(pairs)
            time.sleep(1)

    else:
        inputData, trainTime, dt = train(pairs,predInset,x)
        timeDiff=datetime.now()-trainTime
        B=datetime.now()
      