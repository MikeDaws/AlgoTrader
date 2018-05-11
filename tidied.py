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



def timeout(a):
    time = (a-11.5)/12
    return time





def inOut(inputs, starttraining,endtraining):
    inputPre = inputs[starttraining:endtraining,:]
    
    normInput=[]
    temp=[]
    temp.append(normout(inputPre[:,0:4]))
    temp.append(normout(inputPre[:,6:10]))
    temp.append(normout(inputPre[:,4]))
    temp.append(timeout(inputPre[:,5]))
    temp.append(normout(inputPre[:,10]))
    temp.append(normout(inputPre[:,11]))
    
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
        tempOut.append(inputs[ii+1,0]-inputs[ii,0])
    normOutput = normout(tempOut)
    return normInput, normOutput






def tester(training_y_pred,history1,a,b,c):
    #a is the start index, b is the end
    
    buy=[]
    sell=[]
    trade=[]
    tags=[]
    strong=b
    weak=c
    trade.clear
    
    ignoreVal = training_y_pred.shape[1]
    training_y_pred=training_y_pred.reshape(1,-1,1)
    for ii in range(a,len(training_y_pred[0,:])-1):
#        tags.append(np.sum(training_y_pred[0,ii]))
        if (ii % ignoreVal) != 0:
            
            if np.sum(training_y_pred[0,ii]) > weak:   #weak sell signal
                if (len(sell)>0):
                   trade.append(sell[0]-history1[ii,3])    #if short trade open, then close this
                   sell.clear        
            
            if np.sum(training_y_pred[0,ii]) > strong: #strong buy signal
                if (len(sell)>0):
                   trade.append(sell[0]-history1[ii,3])    #if short trade open, then close this
                   sell.clear
    
                if (len(buy)==0 and len(sell)==0) :
                    buy.append(history1[ii,3]) #no trades open long currency
    
            
            if np.sum(training_y_pred[0,ii]) < -weak:   #weak sell signal
                if (len(buy)>0):
                   trade.append(history1[ii,3]-buy[0])    #if long trade open, then close this
                   buy.clear
                                 
                   
            if np.sum(training_y_pred[0,ii]) < -strong:   #strong sell signal
                
                if (len(buy)>0):
                   trade.append(history1[ii,3]-buy[0])    #if long trade open, then close this
                   buy.clear            
                
                
                if (len(buy)==0 and len(sell)==0) :
                    sell.append(history1[ii,3]) #no trades open short currency
            
    diff=[]
#    for ii in range(a,len(training_y_pred[0,:])-2):
#        diff.append(training_y_pred[0,ii]-history1[ii+1,3])       
    result=np.sum(trade)*10000
    return result,trade,diff

#
    











history1=[]
inputs=[]
##history1=algo.getpast.getpast("EUR_GBP","H1")
history1=algo.getpast.getpast("EUR_USD","H1")
##history1=algo.getpast.getpast("GBP_USD","H1")
##history1=algo.getpast.getpast("USD_JPY","H1")
##history1=algo.getpast.getpast("USD_CAD","H1")

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

inputs.append(talib.abstract.SMA(priceinputs, timeperiod=5))
inputs.append(talib.abstract.SMA(priceinputs, timeperiod=10))
inputs.append(talib.abstract.SMA(priceinputs, timeperiod=50))
inputs.append(talib.abstract.SMA(priceinputs, timeperiod=200))
inputs.append(talib.abstract.ADX(priceinputs))
inputs.append(talib.abstract.RSI(priceinputs))
inputs=np.array(inputs)
inputs=np.transpose(inputs)

starttraining=201
predicted=24

endtraining=len(inputs)-predicted

endtest=len(inputs)-2
seqLength = 24*5
cut =(endtraining-starttraining) % seqLength
adjustedStart=starttraining+cut
starttest=endtest-(endtraining-adjustedStart)    
hidden=100
epochs=600
normIn, normOut = inOut(inputs, adjustedStart, endtraining)    
normInTest, normOutTest = inOut(inputs, starttest, endtest)    

x_batches = normIn.reshape(-1, seqLength, len(inputs[1]))
y_batches = normOut.reshape(-1, seqLength, 1)
x_batches_test = normInTest.reshape(-1, seqLength, len(inputs[1]))
y_batches_test = normOutTest.reshape(-1, seqLength, 1)

output=y_batches.shape[2]
num_periods=y_batches.shape[1]
learning_rate=0.01
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
#for i in range(layers_stacked_count):
#    with tf.variable_scope('RNN_{}'.format(i)):
###        basic_cell.append(tf.nn.rnn_cell.GRUCell(num_units=hidden, activation=tf.nn.relu))

#        LSTMcell=tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden, activation=tf.nn.tanh)
#        basic_cell.append(tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden, activation=tf.nn.tanh))    
#
##basic_cell.append(tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden, activation=tf.nn.sigmoid))    


#basic_cell.append(tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden, activation=tf.nn.sigmoid))
#basic_cell.append(tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden, activation=tf.nn.tanh))       

#LSTMcell=tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden, activation=tf.nn.tanh)
basic_cell.append(tf.nn.rnn_cell.GRUCell(num_units=hidden, activation=tf.nn.tanh))
#basic_cell.append(tf.nn.rnn_cell.GRUCell(num_units=hidden, activation=tf.nn.tanh))
#basic_cell.append(tf.nn.rnn_cell.GRUCell(num_units=hidden, activation=tf.nn.tanh))
#basic_cell.append(tf.nn.rnn_cell.GRUCell(num_units=hidden, activation=tf.nn.tanh))
#basic_cell.append(tf.nn.rnn_cell.GRUCell(num_units=hidden, activation=tf.nn.tanh))


#basic_cell.append(tf.nn.rnn_cell.GRUCell(num_units=hidden, activation=tf.nn.tanh))
#basic_cell.append(tf.contrib.rnn.DropoutWrapper(LSTMcell,input_keep_prob=keep_prob, output_keep_prob=keep_prob))           
#basic_cell.append(tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden, activation=tf.nn.tanh))       
#basic_cell.append(tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden, activation=tf.nn.tanh))       
#basic_cell.append(tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden, activation=tf.nn.tanh))       


#basic_cell.append(tf.contrib.rnn.DropoutWrapper(LSTMcell,input_keep_prob=keep_prob, output_keep_prob=keep_prob))       
#basic_cell.append(tf.contrib.rnn.DropoutWrapper(LSTMcell,input_keep_prob=keep_prob, output_keep_prob=keep_prob))       
#basic_cell.append(tf.contrib.rnn.DropoutWrapper(LSTMcell,input_keep_prob=keep_prob, output_keep_prob=keep_prob))       


#basic_cell.append(tf.contrib.rnn.DropoutWrapper(LSTMcell,input_keep_prob=keep_prob, output_keep_prob=keep_prob))           

basic_cell = tf.nn.rnn_cell.MultiRNNCell(basic_cell, state_is_tuple=True)



#   
#basic_cell = tf.contrib.rnn.LSTMCell(num_units=hidden, activation=tf.nn.tanh)   #create our RNN object
#basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden, activation=tf.nn.relu)
#basic_cell = tf.contrib.rnn.LSTMCell(num_units=hidden, activation=tf.nn.relu)
rnn_output, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)               #choose dynamic over static


stacked_rnn_output = tf.reshape(rnn_output, [-1, hidden])           #change the form into a tensor

#dense = tf.layers.dense(stacked_rnn_output, 100, activation=tf.nn.tanh)

stacked_outputs = tf.layers.dense(stacked_rnn_output, output)        #specify the type of layer (dense)
outputs = tf.reshape(stacked_outputs, [-1, num_periods, output])          #shape of results
 
loss = tf.reduce_sum(tf.square(outputs - y))    #define the cost function which evaluates the quality of our model
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)          #gradient descent method
training_op = optimizer.minimize(loss)          #train the result of the application of the cost_function                                 

init = tf.global_variables_initializer()           #initialize all the variables
saver = tf.train.Saver()

c=[]
testtotal=[]
traintotal=[]
onedayprof=[]
it=[]
with tf.Session() as sess:
    init.run()
    saver.restore(sess, "C:\\New folder\model.ckpt")
    for ep in range(epochs):
        sess.run(training_op, feed_dict={X: x_batches, y: y_batches})
        mse = loss.eval(feed_dict={X: x_batches, y: y_batches})/num_periods
        c.append(mse)

        if ep % 10 == 0:
            
#            print(ep, "Fit:", (mse/num_periods)*100)
            b=mse
            training_y_pred = sess.run(outputs, feed_dict={X: x_batches})
            result_train, end, tags1 = tester(training_y_pred,inputs[adjustedStart:endtraining,:],0,0.5,0)
#            atest1=training_y_pred.reshape(1,-1,1)-y_batches.reshape(1,-1,1)
            training_y_pred1 = sess.run(outputs, feed_dict={X: x_batches_test})
            result_test, endtest1, tags2 = tester(training_y_pred1,inputs[starttest:endtest,:],0,0.5,0)  
            atest1=training_y_pred1.reshape(1,-1,1)-y_batches_test.reshape(1,-1,1)


#            aaa=training_y_pred[0,-24:]
##            bbb=y_batches_test[0,-24:]
##            b1= ((np.sum(((aaa-bbb)**2))))
##            ccc=aaa-bbb
#            b1 = loss.eval(feed_dict={X: x_batches_test, y: y_batches_test})
#            result_test_oneweek, tradeend, tags3 = tester(training_y_pred1,nonnormtest,len(training_y_pred[0,:])-24,0.5,0)
#            y_pred = sess.run(outputs, feed_dict={X: X_test})
#            b =np.sum(abs(outputs-y_batches))

            save_path = saver.save(sess, "C:\\New folder\model.ckpt")
##            onedayprof.append(result_test_oneweek)
#            it.append(ep)
#            
#            
#            ddd=training_y_pred[0,:,0]-y_data[:]       
#            b1= ((np.sum(((training_y_pred[0,:,0]-y_data[:])**2))))/num_periods 
            b2= ((np.sum((abs(training_y_pred1[-1:,-24:,0]-y_batches_test[-1:,-24:,0])))))/24 
            aaa=training_y_pred1[-1:,-24:,0]-y_batches_test[-1:,-24:,0]
            print(ep, "Train:", b)
            print(ep, "Test:", b2)
            print(ep, "Trainprofit", result_train)
#            print(ep, "Testnprofit", result_test)
#            print(ep, "oneweek", result_test_oneweek)
            testtotal.append(b)
            traintotal.append(b2)

            print(ep, "")
#            
##            ddd=y_test-training_y_pred1[0]
#
##plt.plot(aaa)
#            plt.plot(training_y_pred[44,-72:,0]-y_batches[44,-72:,0])
#            plt.plot(training_y_pred[65,-72:,0]-y_batches[65,-72:,0])
#            plt.plot(atest1[0,-24:,0])
#            plt.plot(aaa)
#            plt.plot(training_y_pred1[-1:,-24:,0])
#            plt.plot(y_batches_test[-1:,-24:,0])
#           
#            plt.show()
##            y_pred = sess.run(outputs, feed_dict={Xtest: X_test})
#            plt.figure(figsize=(10,5))
#            plt.title("Forecast vs Actual", fontsize=14)
#            plt.yscale('log')
#            plt.plot(c, "bo", markersize=10, label="Fit")
####plt.plot(pd.Series(np.ravel(Y_test)), "w*", markersize=10)
##            plt.plot(pd.Series(a), "r.", markersize=10, label="Test")
###            plt.legend(loc="upper left")
##            plt.xlabel("Time Periods")
###
#            plt.show()  
#            c=[]
#
##            plt.figure(figsize=(15,10))
##            tt=y_batches[-1:,-num_periods_test:,0]
##
##            tt1=training_y_pred[-1:,-num_periods_test:,0]
##            plt.subplot(211)
##
##            plt.title("Forecast vs Actual", fontsize=14)
##   
##            plt.plot(pd.Series(np.ravel(tt)), "bo", markersize=5, label="Actual")
##            plt.plot(pd.Series(np.ravel(tt1)), "r.", markersize=5, label="Forecast")
###plt.plot(pd.Series(np.ravel(Y_test)), "w*", markersize=10)
##            plt.subplot(212)
##            tt=Y_test[-1:,-num_periods_test:,0]
##
##            tt1=y_pred[-1:,-num_periods_test:,0]           
##            plt.plot(pd.Series(np.ravel(tt)), "bo", markersize=5, label="Actual")
##            plt.plot(pd.Series(np.ravel(tt1)), "r.", markersize=5, label="Forecast")            
##            
##
##            plt.legend(loc="upper left")
##            plt.xlabel("Time Periods")
##            
##            
##            
##
##
##            plt.show()  
#    
#    
#    y_pred = sess.run(outputs, feed_dict={X: X_test})
#
#
#
#
#a= np.sum(abs(y_pred-Y_test))
#
#
##y_pred = sess.run(outputs, feed_dict={X: X_test})
#tt=Y_test[0,-10:,0]
#
#tt1=y_pred[0,-10:,0]
#
#
#
#plt.figure(figsize=(10,5))
#plt.title("Forecast vs Actual", fontsize=14)
#   
#plt.plot(pd.Series(np.ravel(tt)), "bo", markersize=10, label="Actual")
##plt.plot(pd.Series(np.ravel(Y_test)), "w*", markersize=10)
#plt.plot(pd.Series(np.ravel(tt1)), "r.", markersize=10, label="Forecast")
#plt.legend(loc="upper left")
#plt.xlabel("Time Periods")
#
#plt.show()  
#
##fig_size = plt.rcParams["figure.figsize"]
##fig_size[0] = 12
##fig_size[1] = 9
##plt.rcParams["figure.figsize"] = fig_size





