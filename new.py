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
    normout=(a-meancalc)/stdcalc
    return normout




def tester(training_y_pred,history1,a):
    #a is the start index, b is the end
    
    buy=[]
    sell=[]
    trade=[]
    tags=[]
    strong=4
    weak=0.4
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



#meanvol=np.sum(history1[201:,4])/len(history1[201:,4])
#standarddev = (np.sum((history1[201:,4] - meanvol)**2) / len(history1[201:,4]))**0.5
##
#
#
#(history2[201:,1]-np.amin(history2[:,3]))/(np.amax(history2[:,3])-np.amin(history2[:,3]))
##
##(sma5[201:]-np.amin(history2[:,3]))/(np.amax(history2[:,3])-np.amin(history2[:,3]))
#

mean1=np.nanmean(oneback)
std1=np.nanstd(oneback)
norm1=(oneback-mean1)/std1

mean2=np.nanmean(twoback)
std2=np.nanstd(twoback)
norm2=(twoback-mean2)/std2

mean3=np.nanmean(threeback)
std3=np.nanstd(threeback)
norm3=(threeback-mean3)/std3

mean4=np.nanmean(fourback)
std4=np.nanstd(fourback)
norm4=(fourback-mean4)/std4

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
    




mean1=np.nanmean(oneback)
std1=np.nanstd(oneback)
norm1=(oneback-mean1)/std1

mean2=np.nanmean(twoback)
std2=np.nanstd(twoback)
norm2=(twoback-mean2)/std2

mean3=np.nanmean(threeback)
std3=np.nanstd(threeback)
norm3=(threeback-mean3)/std3

mean4=np.nanmean(fourback)
std4=np.nanstd(fourback)
norm4=(fourback-mean4)/std4



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
#history1 = np.vstack((history1,(history2[201:,2]-np.amin(history2[:,3]))/(np.amax(history2[:,3])-np.amin(history2[:,3]))))
#history1 = np.vstack((history1,(history2[201:,3]-np.amin(history2[:,3]))/(np.amax(history2[:,3])-np.amin(history2[:,3]))))
#
##history1 = np.vstack((history1,(history2[201:,4]-np.amin(history2[:,4]))/(np.amax(history2[:,4])-np.amin(history2[:,4]))))
##
#
##history1 = np.vstack((history1,(history2[201:,4]-meanvol)/(standarddev)))
#
#
#history1 = np.vstack((history1,(sma5[201:]-np.amin(history2[:,3]))/(np.amax(history2[:,3])-np.amin(history2[:,3]))))
##history1 = np.vstack((history1,(sma10[201:]-np.amin(history2[:,3]))/(np.amax(history2[:,3])-np.amin(history2[:,3]))))
##history1 = np.vstack((history1,(sma50[201:]-np.amin(history2[:,3]))/(np.amax(history2[:,3])-np.amin(history2[:,3]))))
##history1 = np.vstack((history1,(sma200[201:]-np.amin(history2[:,3]))/(np.amax(history2[:,3])-np.amin(history2[:,3]))))
##history1 = np.vstack((history1,adx[201:]/100))
##history1 = np.vstack((history1,RSI[201:]/100))
##history1 = np.vstack((history1,PLUS_DI[201:]/100))
##history1 = np.vstack((history1,MINUS_DI[201:]/100))
##history1 = np.vstack((history1,(Bol[0][201:]-np.amin(history2[:,3]))/(np.amax(history2[:,3])-np.amin(history2[:,3]))))
##history1 = np.vstack((history1,(Bol[1][201:]-np.amin(history2[:,3]))/(np.amax(history2[:,3])-np.amin(history2[:,3]))))
##history1 = np.vstack((history1,(Bol[2][201:]-np.amin(history2[:,3]))/(np.amax(history2[:,3])-np.amin(history2[:,3]))))
#
##history1 = np.vstack((history1,(history3[201:,0]-np.amin(history3[:,3]))/(np.amax(history3[:,3])-np.amin(history3[:,3]))))
##history1 = np.vstack((history1,(history3[201:,1]-np.amin(history3[:,3]))/(np.amax(history3[:,3])-np.amin(history3[:,3]))))
##history1 = np.vstack((history1,(history3[201:,2]-np.amin(history3[:,3]))/(np.amax(history3[:,3])-np.amin(history3[:,3]))))
##history1 = np.vstack((history1,(history3[201:,3]-np.amin(history3[:,3]))/(np.amax(history3[:,3])-np.amin(history3[:,3]))))
##history1 = np.vstack((history1,(history3[201:,4]-np.amin(history3[:,4]))/(np.amax(history3[:,4])-np.amin(history3[:,4]))))
##history1 = np.vstack((history1,(sma5_2[201:]-np.amin(history3[:,3]))/(np.amax(history3[:,3])-np.amin(history3[:,3]))))
##history1 = np.vstack((history1,(sma10_2[201:]-np.amin(history3[:,3]))/(np.amax(history3[:,3])-np.amin(history3[:,3]))))
##history1 = np.vstack((history1,(sma50_2[201:]-np.amin(history3[:,3]))/(np.amax(history3[:,3])-np.amin(history3[:,3]))))
##history1 = np.vstack((history1,(sma200_2[201:]-np.amin(history3[:,3]))/(np.amax(history3[:,3])-np.amin(history3[:,3]))))
##
##history1 = np.vstack((history1,adx_2[201:]/100))
##history1 = np.vstack((history1,RSI_2[201:]/100))
#
#
#
#
#

##df=pd.read_csv('1hrEURUSD.csv', sep=',',header=0)
##df = df.drop('Open Timestamp', 1)
##df = df.drop('High', 1)
##df = df.drop('Low', 1)
##df = df.drop('Volume', 1)
##df = df.drop('Open', 1)
##df=df.values
##TS = np.array(df)
#
#f_horizon = 1 #forecast horizon, one period into the future
num_periods = TS.shape[0]     #number of periods per vector we are using to predict one period ahead
#num_periods_test = history1.shape[0]-f_horizon 
inputs = 10         #number of vectors submittedxxx
hidden = 100        #number of neurons we will recursively work through, can be changed to improve accuracy
output = 4            #number of output vectors

learning_rate = 0.001    #small learning rate so we don't overshoot the minimum
#tf.layers
epochs = 500    #number of iterations or training cycles, includes both the FeedFoward and Backpropogation
#outputcolumn = 4
layers_stacked_count = 1
##len(history[:,3])/num_periods
#
##TS = history[:,[0,1,2,3]]/np.amax(history[:,3])
##TS1 = history[:,[4]]/np.amax(history[:,4])
#
##TS2 = history[:,2]/np.amax(history[:,2])
##TS = history1 /np.amax(history1[:,1])
##TS = np.hstack((TS1,TS2))
##TS =  np.row_stack((TS[:,1],TS[:,4]))
##TS=np.array(TS[:,1],TS[:,4])
##TS=np.array()
##TS[1,:]=100*(TS[: ,4]-TS[0,4])/TS[0,4]
##TS[:,0]=100*(TS1[: ,4]-TS1[0,4])/TS1[0,4]
##TS[:,1]=100*(TS1[: ,1]-TS1[0,1])/TS1[0,1]
#
##TS[:,1]=100*(TS[: ,4]-TS[0,4])/TS[0,4]
##TS[:,1]=100*(TS1[: ,1]-TS1[0,1])/TS1[0,1]
#
#
##random.seed(111)
##rng = pd.date_range(start='2000', periods=409, freq='M')
##ts = pd.Series(np.random.uniform(-10, 10, size=len(rng)), rng).cumsum()
##ts.plot(c='b', title='Example Time Series')
##plt.show()
##ts.head(10)
#
#
##TS = np.array(ts)
#
#
##x_data = TS[:(len(TS)-((len(TS)-num_periods_test) % num_periods))]
#
#
x_data = TS[:len(TS)]
#
y_data=target1
x_batches = x_data.reshape(-1, num_periods, inputs)
#
#y_data = TS[f_horizon:len(TS),outputcolumn]
y_batches = y_data.reshape(-1, num_periods, output)
#

num_periods_test = TStest.shape[0]
x_test = TStest[:len(TStest)]
x_batches_test = x_test.reshape(-1, num_periods_test, inputs)
y_test=targettest
y_batches_test = y_test.reshape(-1, num_periods_test, output)

#
##x_data = TS[((len(TS)-f_horizon) % num_periods):len(TS)-f_horizon]
##
##x_batches = x_data.reshape(-1, num_periods, inputs)
##
##y_data = TS[((len(TS)) % num_periods):len(TS)-num_periods_test,outputcolumn]
##y_batches = y_data.reshape(-1, num_periods, 1)
##
#
##
##x_data = TS[((len(TS)-f_horizon) % num_periods):len(TS)-f_horizon]
##
##x_batches = x_data.reshape(-1, num_periods, inputs)
##
##y_data = TS[((len(TS)) % num_periods):len(TS)-num_periods_test,outputcolumn]
##y_batches = y_data.reshape(-1, num_periods, 1)
##
#
#
#
#
#
#
#
#
#
#
##print (len(x_batches))
##print (x_batches.shape)
##print (x_batches[0:2])
##print (len(x_data))
##print (len(y_data))
##print (y_batches[0:1])
##print (y_batches.shape)
#
#
#def test_data(series,forecast,num_periods):
#    test_x_setup = TS[-(num_periods + forecast):]
#    testX = test_x_setup[:num_periods].reshape(-1, num_periods, inputs)
#    testY = TS[-(num_periods):,outputcolumn].reshape(-1, num_periods, 1)
#    return testX,testY
#
#X_test, Y_test = test_data(TS,f_horizon,num_periods)
##print (X_test.shape)
##print (X_test)
#

tf.reset_default_graph()   #We didn't have any previous graph objects running, but this would reset the graphs


X = tf.placeholder(tf.float32, [None, num_periods, inputs])   #create variable objects
#Xtest = tf.placeholder(tf.float32, [None, num_periods_test, inputs]) 
y = tf.placeholder(tf.float32, [None, num_periods, output])

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
###        basic_cell.append(tf.nn.rnn_cell.GRUCell(num_units=hidden, activation=tf.nn.tanh))
#        LSTMcell=tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden, activation=tf.nn.tanh)
#        basic_cell.append(tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden, activation=tf.nn.tanh))    
#
##basic_cell.append(tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden, activation=tf.nn.sigmoid))    


#basic_cell.append(tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden, activation=tf.nn.sigmoid))
#basic_cell.append(tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden, activation=tf.nn.tanh))       

#LSTMcell=tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden, activation=tf.nn.tanh)

#basic_cell.append(tf.contrib.rnn.DropoutWrapper(LSTMcell,input_keep_prob=keep_prob, output_keep_prob=keep_prob))           
basic_cell.append(tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden, activation=tf.nn.tanh))       
basic_cell.append(tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden, activation=tf.nn.tanh))       
basic_cell.append(tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden, activation=tf.nn.tanh))       


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
with tf.Session() as sess:
    init.run()
    saver.restore(sess, "C:\\New folder\model.ckpt")
    for ep in range(epochs):
        sess.run(training_op, feed_dict={X: x_batches, y: y_batches})
        mse = loss.eval(feed_dict={X: x_batches, y: y_batches})
        c.append(mse)

        if ep % 10 == 0:
            
#            print(ep, "Fit:", (mse/num_periods)*100)
            b=mse
            training_y_pred = sess.run(outputs, feed_dict={X: x_batches})
            result_train, end, tags1 = tester(training_y_pred,nonnormtraining,0)
            
            training_y_pred = sess.run(outputs, feed_dict={X: x_batches_test})
            result_test, endtest, tags2 = tester(training_y_pred,nonnormtest,0)  
            aaa=training_y_pred[0,-24:]
            bbb=y_batches_test[0,-24:]
            b1= ((np.sum(((aaa-bbb)**2))))
            ccc=aaa-bbb
            b1 = loss.eval(feed_dict={X: x_batches_test, y: y_batches_test})
            result_test_oneweek, tradeend, tags3 = tester(training_y_pred,nonnormtest,len(training_y_pred[0,:])-1-24)
#            y_pred = sess.run(outputs, feed_dict={X: X_test})
#            b =np.sum(abs(outputs-y_batches))

            save_path = saver.save(sess, "C:\\New folder\model.ckpt")
            
            print(ep, "Train:", b)
            print(ep, "Test:", b1)
            print(ep, "Trainprofit", result_train)
            print(ep, "Testnprofit", result_test)
            print(ep, "oneweek", result_test_oneweek)

            print(ep, "")
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



