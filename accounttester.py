# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 20:29:17 2018

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
import getFullpast
import decimal

#import datetime
#while 1:



#Check if trade is already open for pai
pair="EUR_GBP"
#signal="sell"
##risk=0.01
#stopmulti=round(stopmulti,4)
#profitmulti=round(profitmulti,4)
parser = argparse.ArgumentParser()

common.config.add_argument(parser)
args = parser.parse_args()
account_id = args.config.active_account

api = args.config.create_context()
kwargs = {}


response = api.account.get(account_id)
ac=response.get("account", "200")
account1 = Account(ac)
tradesreq=api.trade.list_open(account_id)
trades=tradesreq.body["trades"]
#a=trades[0]

for ii in range(0,len(trades)):
    openPrice=trades[ii].price
    a=ii
    kwargs = {}
    kwargs["instruments"]=trades[ii].instrument #insert command here from trade
    kwargs["since"]=None
    kwargs["includeUnitsAvailable"]="FALSE"
    kwargs["includeHomeConversions"]="TRUE"
    
    response = api.pricing.get(account_id,**kwargs)
    price=response.get("prices", 200)
    bid=price[0].bids[0].price
    ask=price[0].asks[0].price
    spread=ask-bid
    triallingstop=0
    if trades[ii].currentUnits< 0: #i.e. shorting
        if ask<openPrice-1.5*spread:
            triallingstop=spread*0.5
            takeProfit=0
    if trades[ii].currentUnits>0: #i.e. shorting
        if bid>openPrice+1.5*spread:
            triallingstop=spread*0.5
            takeProfit=0
    kwargs={}
    kwargs1={}
    e = abs(decimal.Decimal(str(bid)).as_tuple().exponent)
    triallingstop=round(triallingstop,e)
    if triallingstop>0:
    #    triallingstop=0.0001
    #    kwargs["id"] = trades[ii].id
    #    kwargs["id"] = trades[ii].id
    #    kwargs["instrument"]="GBP_USD"
    #    kwargs["stopLossOnFill"]=v20.transaction.StopLossDetails(**kwargs1)
    #    kwargs1["price"] = 0
    #    kwargs["takeProfitOnFill"]=v20.transaction.TakeProfitDetails(**kwargs1)
        kwargs["takeProfit"]=None
        kwargs["stopLoss"]=None
        kwargs1["distance"] = triallingstop
        kwargs["trailingStopLoss"]=v20.transaction.TrailingStopLossDetails(**kwargs1)   
        api.trade.set_dependent_orders(account_id,trades[ii].id,**kwargs)#    order=api.trade.Trade(**kwargs)
        
