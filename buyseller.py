# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 14:58:06 2018

@author: Mikw
"""
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

buy=[]
sell=[]
trade=[]

for ii in range(201,4990):
    if np.sum(training_y_pred[0,ii]) > 4:
        if (len(buy)==0 and len(sell)==0) :
            buy.append=close[ii]
            
            
    if np.sum(training_y_pred[0,ii]) < 4:   #sell signal
        if (len(buy)==0 and len(sell)==0) :
            sell.append=close[ii] #no trades open short currency
        if (len(buy)>0):
           trade.append=(close[ii]-sell[0])    #if long trade open, then close this