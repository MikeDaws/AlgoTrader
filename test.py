# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 17:51:07 2017

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


history1=algo.getpast.getpast("EUR_USD","H4")

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

history1=history2[201:,0]
history1 = np.vstack((history1,history2[201:,1]))
history1 = np.vstack((history1,history2[201:,2]))
history1 = np.vstack((history1,history2[201:,3]))
history1 = np.vstack((history1,history2[201:,4]))
history1 = np.vstack((history1,sma5[201:]))
history1 = np.vstack((history1,sma10[201:]))
history1 = np.vstack((history1,sma50[201:]))
history1 = np.vstack((history1,sma200[201:]))
history1 = np.transpose(history1)