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


# convert time series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# transform series into train and test sets for supervised learning
def prepare_data(series, n_test, n_lag, n_seq):
	# extract raw values
	raw_values = series.values
#	raw_values = raw_values.reshape(len(raw_values), 1)
	# transform into supervised learning problem X, y
	supervised = series_to_supervised(raw_values, n_lag, n_seq)
	supervised_values = supervised.values
	# split into train and test sets
	train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
	return train, test




history1=algo.getpast.getpast("EUR_USD","H4")


history3=algo.getpast.getpast("GBP_AUD","H4")

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

history1=history2[201:,0]/np.amax(history2[:,3])
history1 = np.vstack((history1,history2[201:,1]/np.amax(history2[:,3])))
history1 = np.vstack((history1,history2[201:,2]/np.amax(history2[:,3])))
history1 = np.vstack((history1,history2[201:,3]/np.amax(history2[:,3])))
history1 = np.vstack((history1,history2[201:,4]/np.amax(history2[:,4])))
history1 = np.vstack((history1,sma5[201:]/np.amax(history2[:,3])))
history1 = np.vstack((history1,sma10[201:]/np.amax(history2[:,3])))
history1 = np.vstack((history1,sma50[201:]/np.amax(history2[:,3])))
history1 = np.vstack((history1,sma200[201:]/np.amax(history2[:,3])))

history1 = np.vstack((history1,adx[201:]/100))
history1 = np.vstack((history1,RSI[201:]/100))

history1 = np.vstack((history1,history3[201:,0]/np.amax(history3[:,3])))
history1 = np.vstack((history1,history3[201:,1]/np.amax(history3[:,3])))
history1 = np.vstack((history1,history3[201:,2]/np.amax(history3[:,3])))
history1 = np.vstack((history1,history3[201:,3]/np.amax(history3[:,3])))
history1 = np.vstack((history1,history3[201:,4]/np.amax(history3[:,4])))
history1 = np.vstack((history1,sma5_2[201:]/np.amax(history3[:,3])))
history1 = np.vstack((history1,sma10_2[201:]/np.amax(history3[:,3])))
history1 = np.vstack((history1,sma50_2[201:]/np.amax(history3[:,3])))
history1 = np.vstack((history1,sma200_2[201:]/np.amax(history3[:,3])))

history1 = np.vstack((history1,adx_2[201:]/100))
history1 = np.vstack((history1,RSI_2[201:]/100))





history1 = np.transpose(history1)
TS=history1

test = DataFrame(TS)


data=test
n_lag=10
n_seq =2
n_test = 10
dropnan = True


train, test = prepare_data(data, n_test, n_lag, n_seq)

# make forecasts
#forecasts = make_forecasts(train, test, n_lag, n_seq)



X, y = train[:, 0:n_lag], train[:, n_lag:]
X = X.reshape(X.shape[0], 1, X.shape[1])
