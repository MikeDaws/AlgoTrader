# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 09:13:53 2018

@author: Mikw
"""

from datetime import datetime, timedelta
import time
from notest import notest
from notraining import notraining
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
import v20
import argparse
from account.account import Account
from order.args import OrderArguments

#import datetime
#while 1:


#def modifyOrder(pair, signal):

parser = argparse.ArgumentParser()   
common.config.add_argument(parser)
args = parser.parse_args()
account_id = args.config.active_account  
api = args.config.create_context()
kwargs = {}  
response = api.account.get(account_id)
ac=response.get("account", "200")
account1 = Account(ac)

kwargs = {}
kwargs1 = {}
#kwargs1["price"] =     
#kwargs["id"]="282"
#    kwargs["units"]=10
#    kwargs["type"]="MARKET"
#kwargs["stopLoss"]=v20.transaction.StopLossDetails(**kwargs1)
#kwargs1["price"] = None
kwargs["takeProfit"]=None
#kwargs1["distance"] = None
#kwargs["trailingStopLoss"]=v20.transaction.TrailingStopLossDetails(**kwargs1)    
#kwargs["trailingStopLoss"]=None
#marketOrderArgs.parse_arguments(args)

order=api.trade.set_dependent_orders(account_id,2656,**kwargs)