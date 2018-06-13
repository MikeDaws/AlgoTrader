# -*- coding: utf-8 -*-
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
import argparse
import common.config
import common.args

from pandas import DataFrame
from pandas import concat

import tensorflow as tf
import shutil
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.layers as tflayers
from tensorflow.contrib.learn.python.learn import learn_runner
import tensorflow.contrib.metrics as metrics
import tensorflow.contrib.rnn as rnn
from account.account import Account
from order.args import OrderArguments

#Main File for doing shit
#import panda
import talib
import algo.get
import algo.getpast
import common.config
import common.args
import datetime
import getFullpast
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

#def timeout(a):
#    time = (a-11.5)/12
#    return time
def timeout1(a):
    time = np.sin(2*np.pi*a/1440)
    return time

def timeout2(a):
    time = np.cos(2*np.pi*a/1440)
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
        temp.append(timeout1(inputPre[:,loopNum*jj+3]))
        temp.append(timeout2(inputPre[:,loopNum*jj+3]))
        temp.append(hundred(inputPre[:,loopNum*jj+4]))
        temp.append(hundred(inputPre[:,loopNum*jj+5]))
        temp.append(hundred(inputPre[:,loopNum*jj+6]))
        temp.append(normout(inputPre[:,loopNum*jj+7]))
        temp.append(normout(inputPre[:,loopNum*jj+8]))
        temp.append(normout(inputPre[:,loopNum*jj+9]))
        temp.append(normout(inputPre[:,loopNum*jj+10]))
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

def inOut(inputs, starttraining,endtraining, currencyPairs,history1,spread):
    inputPre = inputs[starttraining:endtraining,:]
    
    normInput=[]
    temp=[]
    loopNum=int(len(inputPre[2])/currencyPairs)
    for jj in range(0,(currencyPairs)):
        
        temp.append(normout(inputPre[:,loopNum*jj+0]))       
        temp.append(normout(inputPre[:,loopNum*jj+1])) 
#        temp.append(normout(inputPre[:,loopNum*jj+0:loopNum*jj+4]))
        temp.append(normout(inputPre[:,loopNum*jj+2]))
        temp.append(timeout1(inputPre[:,loopNum*jj+3]))
        temp.append(timeout2(inputPre[:,loopNum*jj+3]))
        temp.append(normout(inputPre[:,loopNum*jj+4]))
        temp.append(hundred(inputPre[:,loopNum*jj+5]))
        
        temp.append(hundred(inputPre[:,loopNum*jj+6]))
        temp.append(normout(inputPre[:,loopNum*jj+7]))
        temp.append(normout(inputPre[:,loopNum*jj+8]))
        temp.append(normout(inputPre[:,loopNum*jj+9]))
        temp.append(normout(inputPre[:,loopNum*jj+10]))
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
        if inputs[ii+1,4]-inputs[ii,4]>2*spread:
            tempOut.append(1)
            tempOut.append(0)
            tempOut.append(0)
    
        elif inputs[ii+1,4]-inputs[ii,4]<-2*spread:
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







learning_rate=0.025
hidden=6
layers_stacked_count=1
predAverage=2
epochs=200
predicted=100
seqLength = 100
starttraining=50
output=3
num_classes=3
beta=0.000 
history1=[]
inputs=[]
dataAmount=50000
history=[]
pair="EUR_GBP"
#history.append(algo.getpast.getpast("GBP_USD","M15"))
#history.append(algo.getpast.getpast("AUD_JPY","H1"))
history.append(getFullpast.getFullpast(pair,"M15",11))
#history.append(algo.getpast.getpast("AUD_JPY","H1"))
#history.append(algo.getpast.getpast("AUD_USD","H1"))
#history.append(algo.getpast.getpast("USD_JPY","H4"))
#history.append(algo.getpast.getpast("USD_CAD","M15"))
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
parser = argparse.ArgumentParser()

common.config.add_argument(parser)
args = parser.parse_args()
account_id = args.config.active_account

api = args.config.create_context()
kwargs = {}


response = api.account.get(account_id)
ac=response.get("account", "200")
account1 = Account(ac)
#account1.dump()
accountBalance=account1.details.balance
tradeCount=account1.details.openTradeCount
trades=account1.position_get(pair)
#    sellTrades=trades.short.tradeIDs
#    buyTrades=trades.long.tradeIDs

#if trades.short.units<0 or trades.long.units>0:
kwargs = {}
kwargs["instruments"]=pair
kwargs["since"]=None
kwargs["includeUnitsAvailable"]="FALSE"
kwargs["includeHomeConversions"]="TRUE"


#kwargs["granularity"] = "M15"
#kwargs["count"] = 1
response = api.pricing.get(account_id,**kwargs)
price=response.get("prices", 200)
bid=price[0].bids[0].price
ask=price[0].asks[0].price
#spread=ask-bid
#    spread=0.00015

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
    inputs.append(history1[:,1]-history1[:,3])
    inputs.append(history1[:,1]-history1[:,0])
    inputs.append(history1[:,2]-history1[:,3])
    inputs.append(history1[:,2]-history1[:,0])
    
    upperMovement=history1[1:339998,1]-history1[0:339997,3]
    lowerMovement=history1[1:339998,1]-history1[0:339997,3]
    
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


endtraining=len(inputs)-(predicted+1)

endtest=len(inputs)-1

cut =(endtraining-starttraining) % seqLength
adjustedStart=starttraining+cut
starttest=endtest-(endtraining-adjustedStart)    

normIn, normOut = inOut(inputs, adjustedStart, endtraining, len(history),targetHistory,spread)    
indicatorVal=len(inputs)-(endtraining-adjustedStart)
normindVal = inOnly(inputs, indicatorVal, len(inputs), len(history))    
x_batches_nonnorm = inputs[adjustedStart:endtraining,:].reshape(-1, seqLength, len(inputs[1]))
#x_ind = normindVal.reshape(-1, seqLength, len(normIn[1]))

normIn=normIn[-dataAmount:,:]
normOut=normOut[-dataAmount:,:]
historyUsed=history1[-dataAmount:,:]

x_batches = normIn.reshape(-1, seqLength, len(normIn[1]))
y_batches = normOut.reshape(-1, seqLength, num_classes)



#x_batches_test = normInTest.reshape(-1, seqLength, len(normIn[1]))
#y_batches_test = normOutTest.reshape(-1, seqLength, 1)
testData=normOut[-100:,:]
output=y_batches.shape[2]
num_periods=y_batches.shape[1]
tf.reset_default_graph()   #We didn't have any previous graph objects running, but this would reset the graphs


X = tf.placeholder(tf.float32, [None, x_batches.shape[1], x_batches.shape[2]])   #create variable objects
#Xtest = tf.placeholder(tf.float32, [None, num_periods_test, inputs]) 
y = tf.placeholder(tf.float32, [None, y_batches.shape[1], y_batches.shape[2]])


keep_prob=1
basic_cell = []
for i in range(layers_stacked_count):
#    GruCell=tf.nn.rnn_cell.GRUCell(num_units=hidden, activation=tf.nn.tanh)
#    dropped=tf.contrib.rnn.DropoutWrapper(GruCell,input_keep_prob=keep_prob, output_keep_prob=keep_prob)      
    LSTMCell=tf.nn.rnn_cell.LSTMCell(num_units=hidden, activation=tf.nn.tanh)
#    dropped=tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(LSTMCell)
    dropped=tf.contrib.rnn.DropoutWrapper(LSTMCell,input_keep_prob=keep_prob, output_keep_prob=keep_prob)     
#    dropped=tf.contrib.rnn.DropoutWrapper(GruCell,input_keep_prob=keep_prob, output_keep_prob=keep_prob)
#    basic_cell.append(cudaLSTM)
    basic_cell.append(dropped)

       

basic_cell = tf.nn.rnn_cell.MultiRNNCell(basic_cell, state_is_tuple=True)



rnn_output, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)               #choose dynamic over static
#rnn_output, states ==tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=2,num_units=hidden)  

stacked_rnn_output = tf.reshape(rnn_output, [-1, hidden])           #change the form into a tensor
#dense = tf.layers.dense(stacked_rnn_output, 3, activation=tf.nn.softmax)
stacked_outputs = tf.layers.dense(stacked_rnn_output, output)        #specify the type of layer (dense)
outputs = tf.reshape(stacked_outputs, [-1, num_periods, output]) 
#softmax_w = tf.get_variable("softmax_w", [hidden, num_classes])
#softmax_b = tf.get_variable("softmax_b", [num_classes])
#
#output_logits = tf.add(tf.matmul(stacked_rnn_output,softmax_w),softmax_b)
#output_all = output_logits
#prediction=tf.nn.softmax(output_logits)
#output_reshaped = tf.reshape(output_all,[-1,num_periods, num_classes])
#output_last = tf.gather(tf.transpose(output_reshaped,[1,0,2]), num_periods - 1) 


#dense = tf.layers.dense(stacked_rnn_output, 100, activation=tf.nn.tanh)

#stacked_outputs = tf.layers.dense(stacked_rnn_output, output)        #specify the type of layer (dense)
#outputs = tf.reshape(stacked_outputs, [-1, num_periods, output])          #shape of results
 

adam = tf.contrib.opt.LazyAdamOptimizer(learning_rate=learning_rate)

net = [v for v in tf.trainable_variables()]
weight_reg = tf.add_n([beta * tf.nn.l2_loss(var) for var in net]) #L2
#weight_reg =0
loss1 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=outputs, labels=y_batches, name = 'softmax')    #define the cost function which evaluates the quality of our model
loss=tf.reduce_mean(loss1)+weight_reg
#gradients = adam.compute_gradients(loss)
#clipped_gradients = [(tf.clip_by_value(grad, -0.5, 0.5), var) for grad, var in gradients]
#training_op = adam.apply_gradients(clipped_gradients)
#training_op =tf.contrib.opt.LazyAdamOptimizer(learning_rate=learning_rate)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)          #gradient descent method
training_op = optimizer.minimize(loss) 
         #train the result of the application of the cost_function                                 
#correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()           #initialize all the variables
saver = tf.train.Saver()
tempOut1=[]
 
c=[]
testtotal=[]
indvaltest=[]
resultCash=[]
traintotal=[]
onedayprof=[]
it=[]
testlogger=[]
tot=0
correct_pred=[1]
testtot=[]
with tf.Session() as sess:
    init.run()
#    saver.restore(sess, "C:\\New folder\model.ckpt")
    
    for ep in range(epochs):
#    ep=0
#    while (tot/len(correct_pred)<0.99):
#        ep=ep+1
        loss12 =sess.run(training_op, feed_dict={X: x_batches, y: y_batches})
        mse = loss.eval(feed_dict={X: x_batches, y: y_batches})/num_periods
        c.append(mse)
#        print("Step " + str(ep) + ", Minibatch Loss= " + \
#                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
#                  "{:.3f}".format(acc))
            
        if ep % 10== 0:
            training_y_pred = sess.run(outputs, feed_dict={X: x_batches})
#            print(ep, "Train (MAE):",mse)
#            print(ep, "Train (MAE):",acc)

            save_path = saver.save(sess, "C:\\New folder\model.ckpt")
            
            training_y_pred=training_y_pred.reshape(1,-1,num_classes)
            training_y_pred=np.squeeze(training_y_pred)
            training_y_pred=tf.Tensor.eval(tf.nn.softmax(training_y_pred))
            correct_pred = np.equal(np.argmax(training_y_pred, 1), np.argmax(normOut, 1))         
            tot=np.sum(correct_pred)
            print(ep, "Train (MAE):",tot/len(correct_pred))
            traintotal.append(tot/len(correct_pred))
         
#            for ii in range(1,predicted+1):
#            for ii in range(1,predicted+1):
            ii=seqLength
            normInTest, normOutTest = inOut(inputs, adjustedStart, endtraining+ii, 1,targetHistory,spread) 
            normInTest=normInTest[-dataAmount:,:]
            normOutTest=normOutTest[-dataAmount:,:]
            x_batches_test = normInTest.reshape(-1, seqLength, len(normInTest[1]))
            y_batches_test = normOutTest.reshape(-1, seqLength, 1)
            training_y_pred1 = sess.run(outputs, feed_dict={X: x_batches_test})                
            training_y_pred1=training_y_pred1.reshape(1,-1,num_classes)
            training_y_pred1=np.squeeze(training_y_pred1)
            training_y_pred1=tf.Tensor.eval(tf.nn.softmax(training_y_pred1))
            correct_pred1 = np.equal(np.argmax(training_y_pred1[-seqLength:], 1), np.argmax(normOutTest[-seqLength:], 1))
            correct_test = np.argmax(training_y_pred1[-seqLength:], 1), np.argmax(normOutTest[-seqLength:], 1)
#            testtot.append(correct_pred1[-1])
    #                print(ep, "Train (MAE):",tot/len(predicted))   

            testPredLog=training_y_pred1[-100:,:]
            testLog=normOutTest[-100:,:]
         
#            testtot=np.array(testtot)
            testtotval = np.sum(correct_pred1)
            print(ep, "Test :",testtotval/seqLength)
            testlogger.append(testtotval/seqLength)
                       
            np.sum(normOutTest[:,2])/len(normOutTest[:,2])
#            plt.plot(traintotal, label="Train")
#            plt.plot(testlogger, label="Test")
#            plt.legend(loc="upper left")
