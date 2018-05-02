# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 11:42:27 2018

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