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


history1=algo.getpast.getpast("EUR_USD","H1")

#output = SMA(history1[0,:], timeperiod=25)

inputs = {
    'open': history1[:,0],
    'high': history1[:,1],
    'low': history1[:,2],
    'close': history1[:,3],
    'volume': history1[:,4]
    }

#A=history1[:,0]

sma5 = talib.abstract.SMA(inputs, timeperiod=5)
sma10 = talib.abstract.SMA(inputs, timeperiod=10)

sma50 = talib.abstract.SMA(inputs, timeperiod=50)

sma200 = talib.abstract.SMA(inputs, timeperiod=200)

adx = talib.abstract.ADX(inputs)
RSI = talib.abstract.RSI(inputs)

history2=history1

history1=history2[201:,0]/np.amax(history2[:,3])
history1 = np.vstack((history1,history2[201:,1]/np.amax(history2[:,3])))
history1 = np.vstack((history1,history2[201:,2]/np.amax(history2[:,3])))
history1 = np.vstack((history1,history2[201:,3]/np.amax(history2[:,3])))
history1 = np.vstack((history1,history2[201:,4]/np.amax(history2[:,4])))
history1 = np.vstack((history1,sma5[201:]/np.amax(history2[:,3])))
history1 = np.vstack((history1,sma10[201:]/np.amax(history2[:,3])))
history1 = np.vstack((history1,sma50[201:]/np.amax(history2[:,3])))
history1 = np.vstack((history1,sma200[201:]/np.amax(history2[:,3])))

history1 = np.vstack((history1,adx[201:]/100))

history1 = np.vstack((history1,RSI[201:]/100))


history1 = np.transpose(history1)
TS=history1
#df=pd.read_csv('1hrEURUSD.csv', sep=',',header=0)
#df = df.drop('Open Timestamp', 1)
#df = df.drop('High', 1)
#df = df.drop('Low', 1)
#df = df.drop('Volume', 1)
#df = df.drop('Open', 1)
#df=df.values
#TS = np.array(df)

num_periods = 200    #number of periods per vector we are using to predict one period ahead
num_periods_test = 12
inputs = 11         #number of vectors submittedxxx
hidden = 100          #number of neurons we will recursively work through, can be changed to improve accuracy
output = 1            #number of output vectors
f_horizon = 1  #forecast horizon, one period into the future
learning_rate = 0.001   #small learning rate so we don't overshoot the minimum
#tf.layers
epochs = 5000     #number of iterations or training cycles, includes both the FeedFoward and Backpropogation
outputcolumn = 3
layers_stacked_count = 2
#len(history[:,3])/num_periods

#TS = history[:,[0,1,2,3]]/np.amax(history[:,3])
#TS1 = history[:,[4]]/np.amax(history[:,4])

#TS2 = history[:,2]/np.amax(history[:,2])
#TS = history1 /np.amax(history1[:,1])
#TS = np.hstack((TS1,TS2))
#TS =  np.row_stack((TS[:,1],TS[:,4]))
#TS=np.array(TS[:,1],TS[:,4])
#TS=np.array()
#TS[1,:]=100*(TS[: ,4]-TS[0,4])/TS[0,4]
#TS[:,0]=100*(TS1[: ,4]-TS1[0,4])/TS1[0,4]
#TS[:,1]=100*(TS1[: ,1]-TS1[0,1])/TS1[0,1]

#TS[:,1]=100*(TS[: ,4]-TS[0,4])/TS[0,4]
#TS[:,1]=100*(TS1[: ,1]-TS1[0,1])/TS1[0,1]


#random.seed(111)
#rng = pd.date_range(start='2000', periods=409, freq='M')
#ts = pd.Series(np.random.uniform(-10, 10, size=len(rng)), rng).cumsum()
#ts.plot(c='b', title='Example Time Series')
#plt.show()
#ts.head(10)


#TS = np.array(ts)


#x_data = TS[:(len(TS)-((len(TS)-num_periods_test) % num_periods))]


x_data = TS[((len(TS)-num_periods_test-f_horizon) % num_periods):len(TS)-num_periods_test-f_horizon]

x_batches = x_data.reshape(-1, num_periods, inputs)

y_data = TS[((len(TS)-num_periods_test) % num_periods):len(TS)-num_periods_test,outputcolumn]
y_batches = y_data.reshape(-1, num_periods, 1)
#print (len(x_batches))
#print (x_batches.shape)
#print (x_batches[0:2])
#print (len(x_data))
#print (len(y_data))
#print (y_batches[0:1])
#print (y_batches.shape)


def test_data(series,forecast,num_periods):
    test_x_setup = TS[-(num_periods + forecast):]
    testX = test_x_setup[:num_periods].reshape(-1, num_periods, inputs)
    testY = TS[-(num_periods):,outputcolumn].reshape(-1, num_periods, 1)
    return testX,testY

X_test, Y_test = test_data(TS,f_horizon,num_periods_test)
print (X_test.shape)
print (X_test)
