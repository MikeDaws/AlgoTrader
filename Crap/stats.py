
"""
Created on Sun May 27 09:54:07 2018

@author: Mikw
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 17:38:45 2017

@author: Mikw
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 10:53:29 2017

@author: Mikw
"""

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
import datetime
tf.__version__
#from talib.abstract import *

#Parameter list
loadPrev = True
trainOn = False
filePath = "C:/"


def normout(a):
    meancalc=np.nanmean(a)
    stdcalc=np.nanstd(a)
    normout=(np.tanh(((a-meancalc)/stdcalc)))
    return normout

def timeout(a):
    time = (a-11.5)/12
    return time

def hundred(a):
    time = (a-50)/100
    return time

def twohundred(a):
    time = (a)/200
    return time
 
def inOnly(inputs, starttraining,endtraining, currencyPairs):
    inputPre = inputs[starttraining:endtraining,:]
    
    normInput=[]
    temp=[]
    loopNum=int(len(inputPre[2])/currencyPairs)
    for jj in range(0,(currencyPairs)):
        
        
        temp.append(normout(inputPre[:,loopNum*jj+0]))       
        temp.append(normout(inputPre[:,loopNum*jj+1])) 
#        temp.append(normout(inputPre[:,loopNum*jj+0:loopNum*jj+4]))
        temp.append(normout(inputPre[:,loopNum*jj+2]))
        temp.append(timeout(inputPre[:,loopNum*jj+3]))
        temp.append(hundred(inputPre[:,loopNum*jj+4]))
        temp.append(hundred(inputPre[:,loopNum*jj+5]))
#        temp.append(normout(inputPre[:,loopNum*jj+0:loopNum*jj+4]))
#        temp.append(normout(inputPre[:,loopNum*jj+4]))
#        temp.append(timeout(inputPre[:,loopNum*jj+5]))
#        temp.append(normout(inputPre[:,loopNum*jj+6:loopNum*jj+8]))
#        temp.append(normout(inputPre[:,loopNum*jj+8]))
#        temp.append(hundred(inputPre[:,loopNum*jj+10]))
#        temp.append(hundred(inputPre[:,loopNum*jj+11]))
#        temp.append(hundred(inputPre[:,loopNum*jj+12]))
#        temp.append(hundred(inputPre[:,loopNum*jj+13]))    
#        temp.append(normout(inputPre[:,loopNum*jj+14:loopNum*jj+17]))
#        temp.append(twohundred(inputPre[:,loopNum*jj+17]))
#        temp.append(normout(inputPre[:,loopNum*jj+18]))
#        temp.append(normout(inputPre[:,loopNum*jj+19:loopNum*jj+21]))
#        temp.append(normout(inputPre[:,loopNum*jj+21]))
#        temp.append(hundred(inputPre[:,loopNum*jj+22:loopNum*jj+26]))    
#        temp.append(normout(inputPre[:,loopNum*jj+26]))  
    

    for ii in range(0,len(temp)):
        if temp[ii].ndim > 1:
            for jj in range(0,len(temp[ii][1])):
                    normInput.append(temp[ii][:,jj])
        
        if temp[ii].ndim == 1:       
            normInput.append(temp[ii])    
    normInput=np.array(normInput)
    normInput=np.transpose(normInput)
    
    return normInput

def inOut(inputs, starttraining,endtraining, currencyPairs,history1):
    inputPre = inputs[starttraining:endtraining,:]
    
    normInput=[]
    temp=[]
    loopNum=int(len(inputPre[2])/currencyPairs)
    for jj in range(0,(currencyPairs)):
        
        temp.append(normout(inputPre[:,loopNum*jj+0]))       
        temp.append(normout(inputPre[:,loopNum*jj+1])) 
#        temp.append(normout(inputPre[:,loopNum*jj+0:loopNum*jj+4]))
        temp.append(normout(inputPre[:,loopNum*jj+2]))
        temp.append(timeout(inputPre[:,loopNum*jj+3]))
        temp.append(normout(inputPre[:,loopNum*jj+4]))
        temp.append(hundred(inputPre[:,loopNum*jj+5]))
        
        temp.append(hundred(inputPre[:,loopNum*jj+6]))
#        temp.append(normout(inputPre[:,loopNum*jj+6:loopNum*jj+8]))
#        temp.append(normout(inputPre[:,loopNum*jj+8]))
#        temp.append(hundred(inputPre[:,loopNum*jj+10]))
#        temp.append(hundred(inputPre[:,loopNum*jj+11]))
#        temp.append(hundred(inputPre[:,loopNum*jj+12]))
#        temp.append(hundred(inputPre[:,loopNum*jj+13]))    
#        temp.append(normout(inputPre[:,loopNum*jj+14:loopNum*jj+17]))
#        temp.append(twohundred(inputPre[:,loopNum*jj+17]))
#        temp.append(normout(inputPre[:,loopNum*jj+18]))
#        temp.append(normout(inputPre[:,loopNum*jj+19:loopNum*jj+21]))
#        temp.append(normout(inputPre[:,loopNum*jj+21]))
#        temp.append(hundred(inputPre[:,loopNum*jj+22:loopNum*jj+26]))    
#        temp.append(normout(inputPre[:,loopNum*jj+26]))    
    for ii in range(0,len(temp)):
        if temp[ii].ndim > 1:
            for jj in range(0,len(temp[ii][1])):
                    normInput.append(temp[ii][:,jj])
        
        if temp[ii].ndim == 1:       
            normInput.append(temp[ii])    
    normInput=np.array(normInput)
    normInput=np.transpose(normInput)
    tempOut=[]
#    mean = np.nanmean(inputPre[:,5])
#    stdcalc = np.nanstd(inputPre[:,0])
    
    mean = np.nanmean(inputs[starttraining+1:endtraining+1,4]-inputs[starttraining:endtraining,4])
    stdcalc = np.nanstd(inputs[starttraining+1:endtraining+1,4]-inputs[starttraining:endtraining,4])
##    
#    for ii in range(starttraining,endtraining):
#        if inputs[ii+1,4]-inputs[ii,4]>0:
#            tempOut.append(1)
#            tempOut.append(0)
#
#        elif inputs[ii+1,4]-inputs[ii,4]<=0:
#            tempOut.append(0)
#            tempOut.append(1)
##        else:
##            tempOut.append(1)
##            tempOut.append(0)
##            tempOut.append(0)
            
            
    for ii in range(starttraining,endtraining):
        if inputs[ii+1,4]-inputs[ii,4]>mean+0.5*stdcalc:
            tempOut.append(1)
            tempOut.append(0)
            tempOut.append(0)

        elif inputs[ii+1,4]-inputs[ii,4]<mean-0.5*stdcalc:
            tempOut.append(0)
            tempOut.append(1)
            tempOut.append(0)

        else:
            tempOut.append(0)
            tempOut.append(0)
            tempOut.append(1)
            
#            
#            
#    for ii in range(starttraining,endtraining):
#        if abs(history1[ii+1,3]-history1[ii,3])>mean+(stdcalc*0.25):
#            tempOut.append(1)
#            tempOut.append(0)
#
#        else:
#            tempOut.append(0)
#            tempOut.append(1)
##        else:
##            tempOut.append(1)
##            tempOut.append(0)
##            tempOut.append(0)
    tempOut=np.array(tempOut)
    tempOut=np.transpose(tempOut)
    tempOut=tempOut.reshape(endtraining-starttraining, 3)
#        normOutput=tempOut
    return normInput, tempOut







learning_rate=0.005
hidden=200
layers_stacked_count=3
predAverage=3
epochs=2000
predicted=24
seqLength = 24
starttraining=50
output=3
num_classes=3
beta=0.08
history1=[]
inputs=[]
history=[]

#history.append(algo.getpast.getpast("GBP_USD","H1"))
#history.append(algo.getpast.getpast("EUR_USD","H1"))
history.append(algo.getpast.getpast("EUR_GBP","M15"))
#history.append(algo.getpast.getpast("AUD_JPY","H1"))
#history.append(algo.getpast.getpast("AUD_USD","H1"))
#history.append(algo.getpast.getpast("USD_JPY","H4"))
#history.append(algo.getpast.getpast("USD_CAD","H1"))
###history.append(algo.ge
###past.getpast("USD_CHF","H4"))
#history.append(algo.getpast.getpast("NZD_USD","H1"))
#history.append(algo.getpast.getpast("EUR_CHF","H1"))
#history.append(algo.getpast.getpast("GBP_JPY","H1"))
#
#history.append(algo.getpast.getpast("SGD_HKD","H4"))
#history.append(algo.getpast.getpast("GBP_HKD","H4"))
#history.append(algo.getpast.getpast("CHF_HKD","H4"))
#history.append(algo.getpast.getpast("BCO_USD","H4"))
#history.append(algo.getpast.getpast("USB10Y_USD","H4"))
#history.append(algo.getpast.getpast("XAU_USD","H4"))
#history.append(algo.getpast.getpast("SPX500_USD","H4"))
#history.append(algo.getpast.getpast("US30_USD","H4"))
#history.append(algo.getpast.getpast("USD_JPY","H4"))
#history1=algo.getpast.getpast("EUR_GBP","H1")/

targetHistory=[]
targetHistory.append(history[0])
targetHistory=np.array(targetHistory)
targetHistory=np.squeeze(targetHistory)
for kk in range(0,len(history)):
#    history1.clear

    history1=history[kk]
    #history1=algo.getpast.getpast("GBP_USD","H1")
    #history1=algo.getpast.getpast("USD_JPY","H1")
    #history1=algo.getpast.getpast("USD_CAD","H1")
    
    priceinputs = {
        'open': history1[:,0],
        'high': history1[:,1],
        'low': history1[:,2],
        'close': history1[:,3],
        'volume': history1[:,4]
        }
    
#    for ii in range(0,len(history1[1])):
#        inputs.append((history1[:,ii]))
#    #    inputs.append((history2[:,ii]))
    
    inputs.append(history1[:,3]-history1[:,0])
    inputs.append(history1[:,1]-history1[:,2])
    inputs.append(history1[:,4])
    inputs.append(history1[:,5])
    if predAverage>1:
        inputs.append(talib.abstract.SMA(priceinputs, timeperiod=predAverage)) # target conditions    
    else:
        inputs.append(history1[:,3])#    inputs.append(talib.abstract.SMA(priceinputs, timeperiod=10))    
#    inputs.append(talib.abstract.SMA(priceinputs, timeperiod=20))
#    inputs.append(talib.abstract.SMA(priceinputs, timeperiod=50)-talib.abstract.SMA(priceinputs, timeperiod=5))
    

    #inputs.append(talib.abstract.SMA(priceinputs, timeperiod=200))
    
    inputs.append(talib.abstract.ADX(priceinputs))
    inputs.append(talib.abstract.RSI(priceinputs))
    
     
inputs=np.array(inputs)
inputs=np.transpose(inputs)


endtraining=len(inputs)-(predicted+1)

endtest=len(inputs)-1

cut =(endtraining-starttraining) % seqLength
adjustedStart=starttraining+cut
starttest=endtest-(endtraining-adjustedStart)    

   
mean = np.nanmean(inputs[starttraining+1:endtraining+1,4]-inputs[starttraining:endtraining,4])
stdcalc = np.nanstd(inputs[starttraining+1:endtraining+1,4]-inputs[starttraining:endtraining,4])
