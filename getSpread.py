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



def getSpread(pair):
    parser = argparse.ArgumentParser()
    
    common.config.add_argument(parser)
    args = parser.parse_args()
    account_id = args.config.active_account
    
    api = args.config.create_context()
    kwargs = {}
    
    
    response = api.account.get(account_id)
    ac=response.get("account", "200")
    account1 = Account(ac)
    account1.dump()

    #    sellTrades=trades.short.tradeIDs
    #    buyTrades=trades.long.tradeIDs
    
    #if trades.short.units<0 or trades.long.units>0:
    kwargs = {}
    kwargs["instruments"]=pair
    kwargs["since"]=None
    kwargs["includeUnitsAvailable"]="FALSE"
    kwargs["includeHomeConversions"]="TRUE"
    
    response = api.pricing.get(account_id,**kwargs)
    price=response.get("prices", 200)
    bid=price[0].bids[0].price
    ask=price[0].asks[0].price
    spread=ask-bid   
    
    return spread