# -*- coding: utf-8 -*-
"""
Created on Sat May 12 11:37:03 2018

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



plt.plot(res1, label=1)
plt.plot(res2, label=2)
plt.plot(res3, label=3)
plt.plot(res4, label=4)

#plt.plot(gru3_5pairs_300, label=3)
#plt.plot(gru4_5pairs_200, label=4)
#plt.plot(gru2_5pairs_200, label=5)
#plt.plot(testtotal3gru100, label=3)
#plt.plot(testtotal4gru50, label=4)
#plt.plot(testtotal3lstm50, label=3)
#plt.plot(testtotal_1gru_100, label=3)
#plt.plot(testtotal_3lstm_50, label=4)
plt.legend()