
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
import getSpread
from trainOpt import trainOpt
from tradeModel import tradeModel
from notraining import notraining


def optFunc(x):
#pair="EUR_USD"
#layers_stacked_count=x[0]
#hidden=x[1]
#beta=0.0001
#threshold=0.35
#spread=0.00015
##stoploss=0.0001    
    pair="EUR_JPY"
    layers_stacked_count=1
    hidden=18
    beta=0.013
    threshold=x[0]
    spread=0.017
    stoploss=x[1]
    predIn=2
    n=x[2]
    m=x[3]
#    beta=x[2]
#    threshold=x[3]
#    spread=0.00015
#    stoploss=x[4]
#    predIn=2
#    n=x[5]
#    m=x[6]
#    layers_stacked_count=x[0]
#    hidden=x[1]   
#    beta=x[0]
#    threshold=x[0]
#    spread=0.00015
#    stoploss=x[1]   
#    hidden=50
#    layers_stacked_count=3
#    beta=0.0001    
#
    print(x[0], x[1], x[2], x[3])
    Data=np.load('./save.npy')
    predictions, Data, testScore = trainOpt(pair, layers_stacked_count,hidden,beta,spread,stoploss, Data,predIn,m)
#    predictions=notraining(Data,pair,1, hidden, layers_stacked_count, beta)
    profit = tradeModel(predictions, Data, threshold,spread, stoploss,n)
    
    return -profit