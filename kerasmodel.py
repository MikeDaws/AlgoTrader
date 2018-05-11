# -*- coding: utf-8 -*-
"""
Created on Sat May  5 08:05:25 2018

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

import keras
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

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM


#Parameter list
loadPrev = True
trainOn = False
filePath = "C:/"

#def normout(a):
#    meancalc=np.nanmean(a)
#    stdcalc=np.nanstd(a)
#    normout=(a-meancalc)/stdcalc
#    return normout


def normout(a):
    meancalc=np.nanmean(a)
    stdcalc=np.nanstd(a)
    normout=(np.tanh(((a-meancalc)/stdcalc)))
    return normout


def tester(training_y_pred,history1,a,b,c):
    #a is the start index, b is the end
    
    buy=[]
    sell=[]
    trade=[]
    tags=[]
    strong=b
    weak=c
    trade.clear
    for ii in range(a,len(training_y_pred[0,:])-1):
        tags.append(np.sum(training_y_pred[0,ii]))
        if np.sum(training_y_pred[0,ii]) > strong: #strong buy signal
            if (len(buy)==0 and len(sell)==0) :
                buy.append(history1[ii,3]) #no trades open long currency
            if (len(sell)>0):
               trade.append(sell[0]-history1[ii,3])    #if short trade open, then close this
               sell.clear
        if np.sum(training_y_pred[0,ii]) > weak:   #weak sell signal
            if (len(sell)>0):
               trade.append(sell[0]-history1[ii,3])    #if short trade open, then close this
               sell.clear
               
               
        if np.sum(training_y_pred[0,ii]) < -strong:   #strong sell signal
            
            if (len(buy)==0 and len(sell)==0) :
                sell.append(history1[ii,3]) #no trades open short currency
           
            if (len(buy)>0):
               trade.append(history1[ii,3]-buy[0])    #if long trade open, then close this
               buy.clear
               
        if np.sum(training_y_pred[0,ii]) < -weak:   #weak sell signal
            if (len(buy)>0):
               trade.append(history1[ii,3]-buy[0])    #if long trade open, then close this
               buy.clear
       
    result=np.sum(trade)*10000*0.07
    return result,trade,tags



#history1=algo.getpast.getpast("EUR_GBP","H1")
#history1=algo.getpast.getpast("EUR_USD","H1")
history1=algo.getpast.getpast("GBP_USD","H1")
#history1=algo.getpast.getpast("USD_JPY","H1")
#history1=algo.getpast.getpast("USD_CAD","H1")


savedata=history1

inputs = {
    'open': history1[:,0],
    'high': history1[:,1],
    'low': history1[:,2],
    'close': history1[:,3],
    'volume': history1[:,4]
    }

##A=history1[:,0]
#
sma5 = talib.abstract.SMA(inputs, timeperiod=5)
sma10 = talib.abstract.SMA(inputs, timeperiod=10)
#
sma50 = talib.abstract.SMA(inputs, timeperiod=50)
#
sma200 = talib.abstract.SMA(inputs, timeperiod=200)
#a=0





starttraining=206
endtraining=4900

endtest=4924
starttest=endtest-(endtraining-starttraining)



history1=[]
history3=[]



history1=sma5
history1 = np.vstack((history1,sma10))
history1 = np.vstack((history1,sma50))
history1 = np.vstack((history1,sma200))
history1 = np.vstack((history1,savedata[:,4]))
history1 = np.vstack((history1,savedata[:,5]))
history1 = np.transpose(history1)



#endtraining=endtraining-starttraining-201
#starttraining=0

nonnormtraining=savedata[starttraining:endtraining,:]

nonnormtest=savedata[starttest:endtest,:]
#nonnormtraining=[]
oneback=[]
twoback=[]
threeback=[]
fourback=[]
for ii in range(starttraining,endtraining):
    oneback.append(history1[ii,0]-history1[ii-1,0])
    twoback.append(history1[ii,0]-history1[ii-2,0])
    threeback.append(history1[ii,0]-history1[ii-3,0])
    fourback.append(history1[ii,0]-history1[ii-4,0])
#    nonnormtraining.append(history1[ii,:])
    
    
    
f1=[]
f2=[]
f3=[]
f4=[]
for ii in range(starttraining,endtraining):
    f1.append(history1[ii+1,0]-history1[ii,0])
    f2.append(history1[ii+2,0]-history1[ii,0])
    f3.append(history1[ii+3,0]-history1[ii,0])
    f4.append(history1[ii+4,0]-history1[ii,0])


norm1=normout(oneback)
norm2=normout(twoback)
norm3=normout(threeback)
norm4=normout(fourback)



history2=history1
open1=normout(history2[:,1])
close1=normout(history2[:,0])
high1=normout(history2[:,2])
low1=normout(history2[:,3])
volume=normout(history2[:,4])
time = (history1[:,5]-11.5)/12



#construct training set
history1=open1[starttraining:endtraining]
history1 = np.vstack((history1,close1[starttraining:endtraining]))
history1 = np.vstack((history1,high1[starttraining:endtraining]))
history1 = np.vstack((history1,low1[starttraining:endtraining]))
history1 = np.vstack((history1,volume[starttraining:endtraining]))
history1 = np.vstack((history1,time[starttraining:endtraining]))
history1 = np.vstack((history1,norm1))
history1 = np.vstack((history1,norm2))
history1 = np.vstack((history1,norm3))
history1 = np.vstack((history1,norm4))



history1 = np.transpose(history1)
TS=history1


norm = normout(f1)
target1 = norm
norm = normout(f2)
target1 = np.vstack((target1,norm))
norm = normout(f3)
target1 = np.vstack((target1,norm))
norm = normout(f4)
target1 = np.vstack((target1,norm))
target1 = np.transpose(target1)



history3=history1
history1=history2
#nonnormtest=[]
oneback=[]
twoback=[]
threeback=[]
fourback=[]
for ii in range(starttest,endtest):
    oneback.append(history1[ii,0]-history1[ii-1,0])
    twoback.append(history1[ii,0]-history1[ii-2,0])
    threeback.append(history1[ii,0]-history1[ii-3,0])
    fourback.append(history1[ii,0]-history1[ii-4,0])
#    nonnormtest.append(history1[ii,:])
    
    
    
    
f1=[]
f2=[]
f3=[]
f4=[]
for ii in range(starttest,endtest):
    f1.append(history1[ii+1,0]-history1[ii,0])
    f2.append(history1[ii+2,0]-history1[ii,0])
    f3.append(history1[ii+3,0]-history1[ii,0])
    f4.append(history1[ii+4,0]-history1[ii,0])
    



norm1=normout(oneback)
norm2=normout(twoback)
norm3=normout(threeback)
norm4=normout(fourback)



historytest=[]
historytest=open1[starttest:endtest]
historytest = np.vstack((historytest,close1[starttest:endtest]))
historytest = np.vstack((historytest,high1[starttest:endtest]))
historytest = np.vstack((historytest,low1[starttest:endtest]))
historytest = np.vstack((historytest,volume[starttest:endtest]))
historytest = np.vstack((historytest,time[starttest:endtest]))
historytest = np.vstack((historytest,norm1))
historytest = np.vstack((historytest,norm2))
historytest = np.vstack((historytest,norm3))
historytest = np.vstack((historytest,norm4))



historytest = np.transpose(historytest)
TStest=historytest

norm = normout(f1)
targettest = norm
norm = normout(f2)
targettest = np.vstack((targettest,norm))
norm = normout(f3)
targettest = np.vstack((targettest,norm))
norm = normout(f4)
targettest = np.vstack((targettest,norm))
targettest = np.transpose(targettest)

history1=history3

#
#f_horizon = 1 #forecast horizon, one period into the future
num_periods = TS.shape[0]     #number of periods per vector we are using to predict one period ahead
#num_periods_test = history1.shape[0]-f_horizon 
inputs = 10         #number of vectors submittedxxx
hidden = 100        #number of neurons we will recursively work through, can be changed to improve accuracy
output = 4            #number of output vectors

learning_rate = 0.005    #small learning rate so we don't overshoot the minimum
#tf.layers
epochs = 1000    #number of iterations or training cycles, includes both the FeedFoward and Backpropogation
#outputcolumn = 4
layers_stacked_count = 1

x_data = TS[:len(TS)]
#
y_data=target1
x_batches = x_data.reshape(num_periods, -1,  inputs)
#
#y_data = TS[f_horizon:len(TS),outputcolumn]
y_batches = y_data.reshape(-1, num_periods, output)
#

num_periods_test = TStest.shape[0]
x_test = TStest[:len(TStest)]
x_batches_test = x_test.reshape(-1, num_periods_test, inputs)
y_test=targettest
y_batches_test = y_test.reshape(-1, num_periods_test, output)

timesteps=num_periods
data_dim = inputs
num_classes=output
## Generate dummy training data
#x_train = np.random.random((1000, timesteps, data_dim))
#y_train = np.random.random((1000, num_classes))
#
## Generate dummy validation data
#x_val = np.random.random((100, timesteps, data_dim))
#y_val = np.random.random((100, num_classes))
#



model = Sequential()
model.add(LSTM(50, return_sequences=False, stateful=True,
               batch_input_shape=(x_batches.shape[0], x_batches.shape[1], x_batches.shape[2])))  # returns a sequence of vectors of dimension 32
#model.add(LSTM(50, return_sequences=True, stateful=True))  # returns a sequence of vectors of dimension 32
#model.add(LSTM(50, stateful=True))  # return a single vector of dimension 32
model.add(Dense(output))
optimizer = keras.optimizers.Nadam(lr=0.01)
model.compile(loss='mean_absolute_error',
              optimizer=optimizer)

# Generate dummy training data
#x_train = np.random.random((1000, TS.shape[0], inputs))
#y_train = np.random.random((1000, num_classes))

# Generate dummy validation data
#x_val = np.random.random((100, timesteps, data_dim))
#y_val = np.random.random((100, num_classes))

history=model.fit(x_batches, y_test, batch_size=TS.shape[0], epochs=5000)

plt.plot(history.history['loss'], label='train')
#pyplot.plot(model.history['val_loss'], label='test')
plt.legend()
plt.show()

score = model.evaluate(x_batches, y_test, batch_size=TS.shape[0])

y_predict=model.predict(x_batches, verbose=1, batch_size=TS.shape[0])
y_diff=y_test-y_predict

np.sum(abs(y_diff))/len(y_diff)

plt.plot(y_diff)
#pyplot.plot(model.history['val_loss'], label='test')
plt.show()
