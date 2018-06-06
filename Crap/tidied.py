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



def denormaliser(a,b):
    meancalc=np.nanmean(a)
    stdcalc=np.nanstd(a)
    absoluteval=np.arctan(b)*stdcalc+meancalc    
    return absoluteval, meancalc

def deaverager(a,b,c,d,e):
    #a is the absolute change in moving average
    #b in the input data
    #c is the column in the input data containing the moving average
    #obtain last moving average and add the prediction onto it
    newmovingaverage = b[-1,c]+a
    
    #now calculate the new absolute value 
    #d is the column of close values in the input data
    #e is the number of values used in SMA
    deaveraged = (newmovingaverage*e)-(np.sum(b[(-e+1):,d]))
    
    return deaveraged




def deaverager_mk2(a,b,c,d,e,f):
    #a is the absolute change in moving average
    #b in the input data
    #c is the column in the input data containing the moving average
    #obtain last moving average and add the prediction onto it
    newmovingaverage = b[f,c]+a # this should give the SSMA at time=t+1
    
    #now calculate the new absolute value 
    #d is the column of close values in the input data
    #e is the number of values used in SMA
    deaveraged = (newmovingaverage*e)-(np.sum(b[f-e+2:f+1,d]))
    
    return deaveraged

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
        temp.append(normout(inputPre[:,loopNum*jj+0:loopNum*jj+4]))
        temp.append(normout(inputPre[:,loopNum*jj+4]))
        temp.append(timeout(inputPre[:,loopNum*jj+5]))
        temp.append(normout(inputPre[:,loopNum*jj+6:loopNum*jj+8]))
        temp.append(normout(inputPre[:,loopNum*jj+8]))
        temp.append(hundred(inputPre[:,loopNum*jj+10]))
        temp.append(hundred(inputPre[:,loopNum*jj+11]))
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

def inOut(inputs, starttraining,endtraining, currencyPairs):
    inputPre = inputs[starttraining:endtraining,:]
    
    normInput=[]
    temp=[]
    loopNum=int(len(inputPre[2])/currencyPairs)
    for jj in range(0,(currencyPairs)):
        temp.append(normout(inputPre[:,loopNum*jj+0:loopNum*jj+4]))
        temp.append(normout(inputPre[:,loopNum*jj+4]))
        temp.append(timeout(inputPre[:,loopNum*jj+5]))
        temp.append(normout(inputPre[:,loopNum*jj+6:loopNum*jj+8]))
        temp.append(normout(inputPre[:,loopNum*jj+8]))
        temp.append(hundred(inputPre[:,loopNum*jj+10]))
        temp.append(hundred(inputPre[:,loopNum*jj+11]))
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
    for ii in range(starttraining,endtraining):
        tempOut.append(inputs[ii+1,6]-inputs[ii,6])
    normOutput = normout(tempOut)
    return normInput, normOutput



def NNtester(predictedTest,actualTest,threshold):
#    predictedTest = predicted.reshape(1,-1,1)
#    predictedTest=predictedTest[0,-24:,0]
#    actualTest = actual.reshape(1,-1,1)
#    actualTest=actual[0,-24:,0]
    score=[]
    for ii in range(0,len(predictedTest)):
        if predictedTest[ii]>threshold: #buy signal
            if actualTest[ii]>0:
                score.append(1) #catches signal
#            elif (actualTest[ii]<0.5 and actualTest[ii] >= 0):
#                score[ii]=1
            if actualTest[ii]<0:
                score.append(0)
        if predictedTest[ii]<-threshold: #buy signal
            if actualTest[ii]<0:
                score.append(1) #catches signal
#            elif (actualTest[ii]<0.5 and actualTest[ii] >= 0):
#                score[ii]=1
            if actualTest[ii]>0:
                score.append(0)
                
    return np.sum(score)/len(score)







learning_rate=0.005
hidden=100
layers_stacked_count=3
predAverage=5
epochs=10000
predicted=24
seqLength = 24*3


history1=[]
inputs=[]
history=[]

history.append(algo.getpast.getpast("GBP_USD","H1"))
#history.append(algo.getpast.getpast("EUR_USD","H1"))
#history.append(algo.getpast.getpast("EUR_GBP","H1"))
#history.append(algo.getpast.getpast("AUD_JPY","H4"))
#history.append(algo.getpast.getpast("AUD_USD","H4"))
#history.append(algo.getpast.getpast("USD_JPY","H4"))
#history.append(algo.getpast.getpast("USD_CAD","H4"))
#history.append(algo.getpast.getpast("USD_CHF","H4"))
#history.append(algo.getpast.getpast("NZD_USD","H4"))
#history.append(algo.getpast.getpast("EUR_CHF","H4"))
#history.append(algo.getpast.getpast("GBP_JPY","H4"))

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
    
    for ii in range(0,len(history1[1])):
        inputs.append((history1[:,ii]))
    #    inputs.append((history2[:,ii]))
    
    
    
    inputs.append(talib.abstract.SMA(priceinputs, timeperiod=predAverage)) # target conditions
    inputs.append(talib.abstract.SMA(priceinputs, timeperiod=10))    
    inputs.append(talib.abstract.SMA(priceinputs, timeperiod=20))
    inputs.append(talib.abstract.SMA(priceinputs, timeperiod=50)-talib.abstract.SMA(priceinputs, timeperiod=5))
    

    #inputs.append(talib.abstract.SMA(priceinputs, timeperiod=200))
    
    inputs.append(talib.abstract.ADX(priceinputs))
    inputs.append(talib.abstract.RSI(priceinputs))
#    inputs.append(talib.abstract.MINUS_DI(priceinputs))
#    inputs.append(talib.abstract.PLUS_DI(priceinputs))
#    macd, macdsignal, macdhist = talib.abstract.MACD(priceinputs)
#    inputs.append(macd)
#    inputs.append(macdhist)
#    inputs.append(macdsignal)
#    inputs.append(talib.abstract.CCI(priceinputs))
#    inputs.append(talib.abstract.ATR(priceinputs))
#    inputs.append(talib.abstract.MAX(priceinputs))
#    inputs.append(talib.abstract.MIN(priceinputs))
#    inputs.append(talib.abstract.OBV(priceinputs))
#    slowk, slowd = talib.abstract.STOCH(priceinputs)
#    fastk, fastd = talib.abstract.STOCHF(priceinputs)
#    inputs.append(slowk)
#    inputs.append(slowd)
#    inputs.append(fastk)
#    inputs.append(fastd)
#    inputs.append(talib.abstract.SMA(priceinputs, timeperiod=50))    
    
inputs=np.array(inputs)
inputs=np.transpose(inputs)

starttraining=300

endtraining=len(inputs)-(predicted+1)

endtest=len(inputs)-1

cut =(endtraining-starttraining) % seqLength
adjustedStart=starttraining+cut
starttest=endtest-(endtraining-adjustedStart)    

normIn, normOut = inOut(inputs, adjustedStart, endtraining, len(history))    
indicatorVal=len(inputs)-(endtraining-adjustedStart)
normindVal = inOnly(inputs, indicatorVal, len(inputs), len(history))    
x_batches_nonnorm = inputs[adjustedStart:endtraining,:].reshape(-1, seqLength, len(inputs[1]))
x_ind = normindVal.reshape(-1, seqLength, len(normIn[1]))
x_batches = normIn.reshape(-1, seqLength, len(normIn[1]))
y_batches = normOut.reshape(-1, seqLength, 1)
#x_batches_test = normInTest.reshape(-1, seqLength, len(normIn[1]))
#y_batches_test = normOutTest.reshape(-1, seqLength, 1)

output=y_batches.shape[2]
num_periods=y_batches.shape[1]
tf.reset_default_graph()   #We didn't have any previous graph objects running, but this would reset the graphs


X = tf.placeholder(tf.float32, [None, x_batches.shape[1], x_batches.shape[2]])   #create variable objects
#Xtest = tf.placeholder(tf.float32, [None, num_periods_test, inputs]) 
y = tf.placeholder(tf.float32, [None, y_batches.shape[1], y_batches.shape[2]])

#
#basic_cell1 = tf.contrib.rnn.LSTMCell(num_units=hidden, activation=tf.nn.tanh)   #create our RNN object
##basic_cell2 = tf.contrib.rnn.BasicRNNCell(num_units=hidden, activation=tf.nn.relu)   #create our RNN object
##
#basic_cell2 = tf.contrib.rnn.LSTMCell(num_units=hidden, activation=tf.nn.tanh)   #create our RNN object
#basic_cell3 = tf.contrib.rnn.LSTMCell(num_units=hidden, activation=tf.nn.tanh)   #create our RNN object


#basic_cell=[basic_cell1,basic_cell2, basic_cell3]

keep_prob=0.66
basic_cell = []
for i in range(layers_stacked_count):
#    GruCell=tf.nn.rnn_cell.GRUCell(num_units=hidden, activation=tf.nn.tanh)
#    dropped=tf.contrib.rnn.DropoutWrapper(GruCell,input_keep_prob=keep_prob, output_keep_prob=keep_prob)          
#    basic_cell.append(dropped)
    basic_cell.append(tf.nn.rnn_cell.GRUCell(num_units=hidden, activation=tf.nn.tanh))

         

basic_cell = tf.nn.rnn_cell.MultiRNNCell(basic_cell, state_is_tuple=True)



rnn_output, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)               #choose dynamic over static


stacked_rnn_output = tf.reshape(rnn_output, [-1, hidden])           #change the form into a tensor

#dense = tf.layers.dense(stacked_rnn_output, 100, activation=tf.nn.tanh)

stacked_outputs = tf.layers.dense(stacked_rnn_output, output, activation=tf.nn.tanh)        #specify the type of layer (dense)
outputs = tf.reshape(stacked_outputs, [-1, num_periods, output])          #shape of results
 
loss = tf.reduce_sum(tf.abs(outputs - y))    #define the cost function which evaluates the quality of our model
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)          #gradient descent method
training_op = optimizer.minimize(loss)          #train the result of the application of the cost_function                                 

init = tf.global_variables_initializer()           #initialize all the variables
saver = tf.train.Saver()
tempOut1=[]
for ii in range(adjustedStart,endtraining):
    tempOut1.append(inputs[ii+1,6]-inputs[ii,6]) 
c=[]
testtotal=[]
indvaltest=[]
resultCash=[]
traintotal=[]
onedayprof=[]
it=[]
with tf.Session() as sess:
    init.run()
#    saver.restore(sess, "C:\\New folder\model.ckpt")
    for ep in range(epochs):
        sess.run(training_op, feed_dict={X: x_batches, y: y_batches})
        mse = loss.eval(feed_dict={X: x_batches, y: y_batches})/num_periods
        c.append(mse)

        if ep % 20 == 0:
            
            save_path = saver.save(sess, "C:\\New folder\model.ckpt")
            
#            #       Test 1
#
#
#            b=mse
            
#            result_train, end, tags1 = tester(training_y_pred,inputs[adjustedStart:endtraining,:],0,0.5,0)
##            atest1=training_y_pred.reshape(1,-1,1)-y_batches.reshape(1,-1,1)
#            training_y_pred1 = sess.run(outputs, feed_dict={X: x_batches_test})
#            result_test, endtest1, tags2 = tester(training_y_pred1,inputs[starttest:endtest,:],0,0.5,0)  
#            atest1=training_y_pred1.reshape(1,-1,1)-y_batches_test.reshape(1,-1,1)

#            b2= ((np.sum((abs(training_y_pred1[-1:,-24:,0]-y_batches_test[-1:,-24:,0])))))/24 
#            aaa=training_y_pred1[-1:,-24:,0]-y_batches_test[-1:,-24:,0]
#            print(ep, "Train:", b)
#            print(ep, "Test:", b2)
#            print(ep, "Trainprofit", result_train)
#            testtotal.append(b2)
#            traintotal.append(b)
#            print(ep, "")
#            a1=training_y_pred1[-1:,-24:,0]
#            a2=y_batches_test[-1:,-24:,0]
#            plt.plot(a1[0,:])
#            plt.plot(a2[0,:])          
#            plt.show()

            #       Test 2


            b=mse
            result=[]
            trueVal=[]
            predVal=[]
            absoluteError=[]
            absolute_val1=[]
            trueResults=[]
            trueNonnorm=[]
            training_y_pred = sess.run(outputs, feed_dict={X: x_batches})
            for ii in range(0,predicted):
                normInTest, normOutTest = inOut(inputs, adjustedStart+ii, endtraining+ii, len(history)) 
                x_batches_test = normInTest.reshape(-1, seqLength, len(normInTest[1]))
                y_batches_test = normOutTest.reshape(-1, seqLength, 1)
                training_y_pred1 = sess.run(outputs, feed_dict={X: x_batches_test})                
                result.append(training_y_pred1[-1:,-1:,0]-y_batches_test[-1:,-1:,0])
                trueVal.append(y_batches_test[-1:,-1:,0])
                predVal.append(training_y_pred1[-1:,-1:,0])
                denorm_pred1, meanVal1 = denormaliser(tempOut1,training_y_pred1[-1:,-1:,0])
                absolute_val1.append(deaverager(denorm_pred1,inputs[adjustedStart+ii:endtraining+ii,:],6,3,predAverage))
                trueNonnorm.append(inputs[endtraining+ii+1,3]) 
#                absoluteError.append(abs(absolute_val1[-1,-1,-1] - inputs[endtraining+ii+1,3])) 
                trueResults.append(inputs[endtraining+ii+1,6]-inputs[endtraining+ii,6]) 
 
            absolute_val1=np.array(absolute_val1)

            result=np.array(result)
            result=result[:,0,0]
            trueVal=np.array(trueVal)
            trueVal=trueVal[:,0,0]
            predVal=np.array(predVal)
            predVal=predVal[:,0,0]
            absoluteError=(abs(absolute_val1[:,-1,-1] - trueNonnorm)) 

#            absoluteError=np.array(absoluteError)
#            absoluteError=(absolute_val1[:,-1,-1]-inputs[-predicted:,3])
            absoluteError=np.sum(absoluteError)
            resultCash.append(absoluteError)
#            plt.subplot(211)
#            a1=training_y_pred[2:,-10:,0]
#            a2=y_batches[2:,-10:,0]
            real_pred = sess.run(outputs, feed_dict={X: x_ind})
            norm_pred = real_pred[-1,-1,-1]
            
            y_pred=training_y_pred.reshape(-1)
       
            realVal=[]
            checker=[]
            checker_error=[]
            checker_abs=[]
            SMAcheck=[]
            for jj in range (0, endtraining-adjustedStart):
                denorm_pred, meanVal12 = denormaliser(tempOut1,y_pred[jj])
                checker.append(denorm_pred)
                absolute_val = deaverager_mk2(denorm_pred,inputs,6,3,predAverage,adjustedStart+jj)
                checker_abs.append(absolute_val)
                realVal.append(inputs[adjustedStart+jj+1,3])
                checker_error.append(abs(inputs[adjustedStart+jj+1,3]-absolute_val))
                SMAcheck.append(inputs[adjustedStart+jj+1,6]-inputs[adjustedStart+jj,6])
            
            
            
            checker_error_total=np.sum(checker_error)    
#            denorm_pred_check, meanVal_check = denormaliser(tempOut1,norm_pred)
            
            
#            absolute_val = deaverager(denorm_pred,inputs,6,3,predAverage)
#            absoluteChange = absolute_val - inputs[-1,3] 



#            plt.plot(trueVal)
#            plt.plot(predVal)
#            plt.show()
            plt.plot(absolute_val1[:,-1,-1])
            plt.plot(trueNonnorm[:])
            plt.show()
            plt.plot(realVal[0:20])
            plt.plot(checker_abs[0:20])
            plt.show()

            
#            plt.plot(SMAcheck[0:50])
#            plt.plot(checker[0:50])
#            plt.show()
#            plt.plot(training_y_pred[10,:26])
#            plt.plot(y_batches[10,:26])
#            plt.show()
            
            
            
            indvaltest.append(checker_error_total)
            b2= ((np.sum((abs(result)))))/predicted 
            print(ep, "Train (MAE):", b)
            print(ep, "Test (MAE):", b2)
            score=NNtester(denorm_pred1,trueResults,0)

            print(ep, "Buy/Sell (%):", score*100)
            print(ep, "TargetValue:", absolute_val)
#            print(ep, "PotentialChange:", absoluteChange)
            print(ep, "NormalisedChange:", norm_pred)
            print(ep, "NormGrad:", norm_pred-predVal[-1])
            print(ep, "CheckerError:", checker_error_total)
            print(ep, "TestError:", absoluteError)
            print("")
            testtotal.append(b2)
            traintotal.append(b)