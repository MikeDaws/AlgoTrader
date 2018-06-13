

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


def notest(pairIn,predIn,A,spread, hidden, layers_stacked_count, beta,stoploss,m):
    
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
    
    def inOut(inputs, starttraining,endtraining, currencyPairs,history1,spread,m):
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
            if inputs[ii+1,4]-inputs[ii,4]>m*spread:
                tempOut.append(1)
                tempOut.append(0)
                tempOut.append(0)
        
            elif inputs[ii+1,4]-inputs[ii,4]<-m*spread:
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
    
    
    
    
    
    
    
    learning_rate=0.01
#    hidden=12
#    layers_stacked_count=1
    predAverage=predIn
    if predIn>1:
        epochs=200
    else:
        epochs=500
    predicted=0
    seqLength = 100
    starttraining=50
    output=3
    num_classes=3
    beta=0.001
    history1=[]
    inputs=[]
    history=[]
    
    #history.append(algo.getpast.getpast("GBP_USD","H1"))
    #history.append(algo.getpast.getpast("EUR_USD","H1"))
    history.append(A)
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
        
        priceinputs = {
            'open': history1[:,0],
            'high': history1[:,1],
            'low': history1[:,2],
            'close': history1[:,3],
            'volume': history1[:,4]
            }
            
        inputs.append(history1[:,3]-history1[:,0])
        inputs.append(history1[:,1]-history1[:,2])
        inputs.append(history1[:,4])
        inputs.append(history1[:,5])
        if predAverage>1:
            inputs.append(talib.abstract.SMA(priceinputs, timeperiod=predAverage)) # target conditions    
        else:
            inputs.append(history1[:,3])
        inputs.append(talib.abstract.ADX(priceinputs))
        inputs.append(talib.abstract.RSI(priceinputs))
        inputs.append(history1[:,1]-history1[:,3])
        inputs.append(history1[:,1]-history1[:,0])
        inputs.append(history1[:,2]-history1[:,3])
        inputs.append(history1[:,2]-history1[:,0])        
    inputs=np.array(inputs)
    inputs=np.transpose(inputs)
    
    endtraining=len(inputs)-(1)
    
    cut =(endtraining-starttraining) % seqLength
    adjustedStart=starttraining+cut
    
    normIn, normOut = inOut(inputs, adjustedStart, endtraining, len(history),targetHistory,spread,m)    
    x_batches_nonnorm = inputs[adjustedStart:endtraining,:].reshape(-1, seqLength, len(inputs[1]))
    x_batches = normIn.reshape(-1, seqLength, len(normIn[1]))
    y_batches = normOut.reshape(-1, seqLength, num_classes)
    
    output=y_batches.shape[2]
    num_periods=y_batches.shape[1]
    tf.reset_default_graph()   #We didn't have any previous graph objects running, but this would reset the graphs
    
    
    X = tf.placeholder(tf.float32, [None, x_batches.shape[1], x_batches.shape[2]])   #create variable objects
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
    
    stacked_rnn_output = tf.reshape(rnn_output, [-1, hidden])           #change the form into a tensor
    stacked_outputs = tf.layers.dense(stacked_rnn_output, output)        #specify the type of layer (dense)
    outputs = tf.reshape(stacked_outputs, [-1, num_periods, output]) 
    
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
    
    
    pathName='C:\\New folder\\'
    pair=pairIn
    end='.ckpt'
    full=pathName+pair+str(predIn)+end
    
    with tf.Session() as sess:
        init.run()
    #    saver.restore(sess, "C:\\New folder\model.ckpt")
        for ep in range(epochs):
    #    ep=0
    #    while (tot/len(correct_pred)<0.99):
            ep=ep+1
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
    
                save_path = saver.save(sess, full)
                
                training_y_pred=training_y_pred.reshape(1,-1,num_classes)
                training_y_pred=np.squeeze(training_y_pred)
                training_y_pred=tf.Tensor.eval(tf.nn.softmax(training_y_pred))
                correct_pred = np.equal(np.argmax(training_y_pred, 1), np.argmax(normOut, 1))         
                tot=np.sum(correct_pred)
                print(ep, "Train (MAE):",tot/len(correct_pred))

    return targetHistory