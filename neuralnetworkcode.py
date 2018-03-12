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


history1=algo.getpast.getpast("EUR_GBP","H4")


history3=algo.getpast.getpast("CHF_HKD","H4")

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
PLUS_DI =talib.abstract.PLUS_DI(inputs)
MINUS_DI =talib.abstract.MINUS_DI(inputs)
MACD = talib.abstract.MACD(inputs)
OBV = talib.abstract.OBV(inputs)
RSI = talib.abstract.RSI(inputs)
Bol=talib.abstract.BBANDS(inputs)



inputs1 = {
    'open': history3[:,0],
    'high': history3[:,1],
    'low': history3[:,2],
    'close': history3[:,3],
    'volume': history3[:,4]
    }

#A=history1[:,0]

sma5_2 = talib.abstract.SMA(inputs1, timeperiod=5)
sma10_2 = talib.abstract.SMA(inputs1, timeperiod=10)

sma50_2 = talib.abstract.SMA(inputs1, timeperiod=50)

sma200_2 = talib.abstract.SMA(inputs1, timeperiod=200)

adx_2 = talib.abstract.ADX(inputs1)
RSI_2 = talib.abstract.RSI(inputs1)




history2=history1

#
#(history2[201:,1]-np.amin(history2[:,3]))/(np.amax(history2[:,3])-np.amin(history2[:,3]))
#
#(sma5[201:]-np.amin(history2[:,3]))/(np.amax(history2[:,3])-np.amin(history2[:,3]))

history1=(history2[201:,0]-np.amin(history2[:,3]))/(np.amax(history2[:,3])-np.amin(history2[:,3]))
history1 = np.vstack((history1,(history2[201:,1]-np.amin(history2[:,3]))/(np.amax(history2[:,3])-np.amin(history2[:,3]))))
history1 = np.vstack((history1,(history2[201:,2]-np.amin(history2[:,3]))/(np.amax(history2[:,3])-np.amin(history2[:,3]))))
history1 = np.vstack((history1,(history2[201:,3]-np.amin(history2[:,3]))/(np.amax(history2[:,3])-np.amin(history2[:,3]))))
history1 = np.vstack((history1,(history2[201:,4]-np.amin(history2[:,4]))/(np.amax(history2[:,4])-np.amin(history2[:,4]))))
history1 = np.vstack((history1,(sma5[201:]-np.amin(history2[:,3]))/(np.amax(history2[:,3])-np.amin(history2[:,3]))))
history1 = np.vstack((history1,(sma10[201:]-np.amin(history2[:,3]))/(np.amax(history2[:,3])-np.amin(history2[:,3]))))
history1 = np.vstack((history1,(sma50[201:]-np.amin(history2[:,3]))/(np.amax(history2[:,3])-np.amin(history2[:,3]))))
history1 = np.vstack((history1,(sma200[201:]-np.amin(history2[:,3]))/(np.amax(history2[:,3])-np.amin(history2[:,3]))))
history1 = np.vstack((history1,adx[201:]/100))
history1 = np.vstack((history1,RSI[201:]/100))
history1 = np.vstack((history1,PLUS_DI[201:]/100))
history1 = np.vstack((history1,MINUS_DI[201:]/100))
history1 = np.vstack((history1,(Bol[0][201:]-np.amin(history2[:,3]))/(np.amax(history2[:,3])-np.amin(history2[:,3]))))
history1 = np.vstack((history1,(Bol[1][201:]-np.amin(history2[:,3]))/(np.amax(history2[:,3])-np.amin(history2[:,3]))))
history1 = np.vstack((history1,(Bol[2][201:]-np.amin(history2[:,3]))/(np.amax(history2[:,3])-np.amin(history2[:,3]))))

#history1 = np.vstack((history1,(history3[201:,0]-np.amin(history3[:,3]))/(np.amax(history3[:,3])-np.amin(history3[:,3]))))
#history1 = np.vstack((history1,(history3[201:,1]-np.amin(history3[:,3]))/(np.amax(history3[:,3])-np.amin(history3[:,3]))))
#history1 = np.vstack((history1,(history3[201:,2]-np.amin(history3[:,3]))/(np.amax(history3[:,3])-np.amin(history3[:,3]))))
#history1 = np.vstack((history1,(history3[201:,3]-np.amin(history3[:,3]))/(np.amax(history3[:,3])-np.amin(history3[:,3]))))
#history1 = np.vstack((history1,(history3[201:,4]-np.amin(history3[:,4]))/(np.amax(history3[:,4])-np.amin(history3[:,4]))))
#history1 = np.vstack((history1,(sma5_2[201:]-np.amin(history3[:,3]))/(np.amax(history3[:,3])-np.amin(history3[:,3]))))
#history1 = np.vstack((history1,(sma10_2[201:]-np.amin(history3[:,3]))/(np.amax(history3[:,3])-np.amin(history3[:,3]))))
#history1 = np.vstack((history1,(sma50_2[201:]-np.amin(history3[:,3]))/(np.amax(history3[:,3])-np.amin(history3[:,3]))))
#history1 = np.vstack((history1,(sma200_2[201:]-np.amin(history3[:,3]))/(np.amax(history3[:,3])-np.amin(history3[:,3]))))
#
#history1 = np.vstack((history1,adx_2[201:]/100))
#history1 = np.vstack((history1,RSI_2[201:]/100))





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

num_periods = 500    #number of periods per vector we are using to predict one period ahead
num_periods_test = 500
inputs = 16         #number of vectors submittedxxx
hidden = 100         #number of neurons we will recursively work through, can be changed to improve accuracy
output = 1            #number of output vectors
f_horizon = 1 #forecast horizon, one period into the future
learning_rate = 0.005   #small learning rate so we don't overshoot the minimum
#tf.layers
epochs = 3000     #number of iterations or training cycles, includes both the FeedFoward and Backpropogation
outputcolumn = 5
layers_stacked_count = 1
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


x_data = TS[:(len(TS)-((len(TS)-num_periods_test) % num_periods))]


x_data = TS[((len(TS)-num_periods_test-f_horizon) % num_periods):len(TS)-num_periods_test-f_horizon]

x_batches = x_data.reshape(-1, num_periods, inputs)

y_data = TS[((len(TS)-num_periods_test) % num_periods):len(TS)-num_periods_test,outputcolumn]
y_batches = y_data.reshape(-1, num_periods, 1)


#x_data = TS[((len(TS)-f_horizon) % num_periods):len(TS)-f_horizon]
#
#x_batches = x_data.reshape(-1, num_periods, inputs)
#
#y_data = TS[((len(TS)) % num_periods):len(TS)-num_periods_test,outputcolumn]
#y_batches = y_data.reshape(-1, num_periods, 1)
#

#
#x_data = TS[((len(TS)-f_horizon) % num_periods):len(TS)-f_horizon]
#
#x_batches = x_data.reshape(-1, num_periods, inputs)
#
#y_data = TS[((len(TS)) % num_periods):len(TS)-num_periods_test,outputcolumn]
#y_batches = y_data.reshape(-1, num_periods, 1)
#










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

X_test, Y_test = test_data(TS,f_horizon,num_periods)
print (X_test.shape)
print (X_test)


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

keep_prob=0.8
basic_cell = []
#for i in range(layers_stacked_count):
#    with tf.variable_scope('RNN_{}'.format(i)):
###        basic_cell.append(tf.nn.rnn_cell.GRUCell(num_units=hidden, activation=tf.nn.relu))
###        basic_cell.append(tf.nn.rnn_cell.GRUCell(num_units=hidden, activation=tf.nn.tanh))
###        LSTMcell=tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden, activation=tf.nn.tanh)
#        basic_cell.append(tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden, activation=tf.nn.tanh))    
###        basic_cell.append(tf.contrib.rnn.DropoutWrapper(LSTMcell,input_keep_prob=keep_prob, output_keep_prob=keep_prob))           
#
##basic_cell.append(tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden, activation=tf.nn.sigmoid))    

basic_cell.append(tf.nn.rnn_cell.GRUCell(num_units=hidden, activation=tf.nn.relu))       
basic_cell = tf.nn.rnn_cell.MultiRNNCell(basic_cell, state_is_tuple=True)



#   
#basic_cell = tf.contrib.rnn.LSTMCell(num_units=hidden, activation=tf.nn.tanh)   #create our RNN object
#basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden, activation=tf.nn.relu)
#basic_cell = tf.contrib.rnn.LSTMCell(num_units=hidden, activation=tf.nn.relu)
rnn_output, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)               #choose dynamic over static


stacked_rnn_output = tf.reshape(rnn_output, [-1, hidden])           #change the form into a tensor
stacked_outputs = tf.layers.dense(stacked_rnn_output, output)        #specify the type of layer (dense)
outputs = tf.reshape(stacked_outputs, [-1, num_periods, output])          #shape of results
 
loss = tf.reduce_sum(tf.square(outputs - y))    #define the cost function which evaluates the quality of our model
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)          #gradient descent method
training_op = optimizer.minimize(loss)          #train the result of the application of the cost_function                                 

init = tf.global_variables_initializer()           #initialize all the variables


with tf.Session() as sess:
    init.run()
    for ep in range(epochs):
        sess.run(training_op, feed_dict={X: x_batches, y: y_batches})
        if ep % 100 == 0:
            mse = loss.eval(feed_dict={X: x_batches, y: y_batches})
            
#            print(ep, "Fit:", (mse/num_periods)*100)
            b=(mse/num_periods)*100
            training_y_pred = sess.run(outputs, feed_dict={X: x_batches})
            
            y_pred = sess.run(outputs, feed_dict={X: X_test})
#            b =np.sum(abs(outputs-y_batches))
            a= ((np.sum((abs(y_pred-Y_test))))/num_periods)*100
            
#            print(ep, "Fit:", b)
            print(ep, "Test:", a)
#            y_pred = sess.run(outputs, feed_dict={Xtest: X_test})
#            plt.figure(figsize=(10,5))
##            plt.title("Forecast vs Actual", fontsize=14)
##   
#            plt.plot(pd.Series(b), "bo", markersize=10, label="Fit")
###plt.plot(pd.Series(np.ravel(Y_test)), "w*", markersize=10)
#            plt.plot(pd.Series(a), "r.", markersize=10, label="Test")
##            plt.legend(loc="upper left")
#            plt.xlabel("Time Periods")
##
#            plt.show()  
#            plt.figure(figsize=(15,10))
#            tt=y_batches[-1:,-num_periods_test:,0]
#
#            tt1=training_y_pred[-1:,-num_periods_test:,0]
#            plt.subplot(211)
#
#            plt.title("Forecast vs Actual", fontsize=14)
#   
#            plt.plot(pd.Series(np.ravel(tt)), "bo", markersize=5, label="Actual")
#            plt.plot(pd.Series(np.ravel(tt1)), "r.", markersize=5, label="Forecast")
##plt.plot(pd.Series(np.ravel(Y_test)), "w*", markersize=10)
#            plt.subplot(212)
#            tt=Y_test[-1:,-num_periods_test:,0]
#
#            tt1=y_pred[-1:,-num_periods_test:,0]           
#            plt.plot(pd.Series(np.ravel(tt)), "bo", markersize=5, label="Actual")
#            plt.plot(pd.Series(np.ravel(tt1)), "r.", markersize=5, label="Forecast")            
#            
#
#            plt.legend(loc="upper left")
#            plt.xlabel("Time Periods")
#            
#            
#            
#
#
#            plt.show()  
    
    
    y_pred = sess.run(outputs, feed_dict={X: X_test})




a= np.sum(abs(y_pred-Y_test))


#y_pred = sess.run(outputs, feed_dict={X: X_test})
tt=Y_test[0,-num_periods:,0]

tt1=y_pred[0,-num_periods:,0]



plt.figure(figsize=(10,5))
plt.title("Forecast vs Actual", fontsize=14)
   
plt.plot(pd.Series(np.ravel(tt)), "bo", markersize=10, label="Actual")
#plt.plot(pd.Series(np.ravel(Y_test)), "w*", markersize=10)
plt.plot(pd.Series(np.ravel(tt1)), "r.", markersize=10, label="Forecast")
plt.legend(loc="upper left")
plt.xlabel("Time Periods")

plt.show()  

#fig_size = plt.rcParams["figure.figsize"]
#fig_size[0] = 12
#fig_size[1] = 9
#plt.rcParams["figure.figsize"] = fig_size