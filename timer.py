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
#import datetime
#while 1:




def train(pairs,predInset):

    b=0
    trainTime=[]
    for p in pairs:
        inputData.append(algo.getpast.getpast(p,"M15"))
        trainTime=datetime.now()
        
    for p in pairs:    
        for predIn in predInset:       
            notest(p,predIn,inputData[b])
            dt = datetime.now() + timedelta(hours=1)
            dt = dt.replace(minute=00, second=00,  microsecond=4)
        b=b+1
    
    return inputData, trainTime, dt





#train models

pairs=['EUR_USD', "EUR_GBP", "GBP_USD", "AUD_JPY", "AUD_USD", "CAD_JPY", "EUR_CHF", "NZD_CAD", "USD_CHF", "USD_JPY", "ZAR_JPY"]
#pairs=['EUR_USD']
inputDataRaw=[]
inputData=[]
predInset=[2]
#predInset=[1]


t=0

#Train neural networks
inputData, trainTime, dt = train(pairs,predInset)


while 1:
    B=datetime.now()
    t= (B.hour - trainTime.hour)*4+(np.floor(B.minute/15) - np.floor(trainTime.minute/15)) #time difference between now and training
    t=int(t)
    if t<100:
        inputDataRaw=[]       
        #get latest candles      
        a=0
        for p in pairs:
            inputDataRaw.append(algo.getpast.getpast(p,"M15"))  

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
                A=5100-len(temp)
                temp=np.concatenate((temp,np.zeros((A,6))))
        #        inputData[a]=temp
                results=(notraining(temp,p,predIn))
                results=results[-100+int(t),:]
                totResults.append(results)

                np.savetxt("C:\\Users\\Mikw\\Google Drive\\forex\\"+p+"_"+str(predIn)+".csv", results, delimiter=",",fmt="%s", newline='\n')
#                np.savetxt("C:\\Users\\Mikw\\Google Drive\\forex\\time.csv", timeLog, delimiter=",",fmt="%s", newline='\n')
            a=a+1
            totResults=np.array(totResults)
            new=np.sum(totResults, axis=0)/len(totResults)
            if new[0]>0.55:
                mO.placeOrder(p,"buy")
            elif new[1]>0.55:   
                mO.placeOrder(p,"sell")
            
        dt = datetime.now()    
        dt = datetime.now() +  timedelta(minutes=(15-dt.minute%15))
        
        dt = dt.replace(second=00,  microsecond=00)
        #        t=t+1
        while datetime.now() < dt:
            time.sleep(1)

    else:
        inputData, trainTime, dt = train(pairs,predInset)
        timeDiff=datetime.now()-trainTime
        B=datetime.now()
      