# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 17:21:15 2018

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


#def placeOrder(pair, signal):

#Check if trade is already open for pai
pair="EUR_GBP"
signal="sell"
risk=0.01
parser = argparse.ArgumentParser()

common.config.add_argument(parser)
args = parser.parse_args()
account_id = args.config.active_account

api = args.config.create_context()
kwargs = {}


response = api.account.get(account_id)
ac=response.get("account", "200")
account1 = Account(ac)
#account1.dump()
accountBalance=account1.details.balance
tradeCount=account1.details.openTradeCount
trades=account1.position_get(pair)
#    sellTrades=trades.short.tradeIDs
#    buyTrades=trades.long.tradeIDs

#if trades.short.units<0 or trades.long.units>0:
kwargs = {}
kwargs["instruments"]=pair
kwargs["since"]=None
kwargs["includeUnitsAvailable"]="FALSE"
kwargs["includeHomeConversions"]="TRUE"


#kwargs["granularity"] = "M15"
#kwargs["count"] = 1
response = api.pricing.get(account_id,**kwargs)
price=response.get("prices", 200)
bid=price[0].bids[0].price
ask=price[0].asks[0].price
spread=ask-bid
mid=(ask+bid)/2
if signal=="buy":
    stoploss=bid-2*spread
    takeProfit=bid+2.5*spread
    potentialLoss=abs(stoploss-bid)
    sign=1
    convert=price[0].quoteHomeConversionFactors.positiveUnits
elif signal=="sell":
    stoploss=ask+2*spread
    takeProfit=ask-2.5*spread
    potentialLoss=abs(stoploss-bid)
    sign=-1
    convert=price[0].quoteHomeConversionFactors.negativeUnits

#    #ac=response.get("price", "200")
#    #candles = response.get("candles", 200)
#    #
#    stoploss=round(stoploss,8)
#    takeProfit=round(takeProfit,8)
#    #   position size * change *conversion factor/traded currency = account balance * risk
#change in value in gbp = unit size*0.0001*change in pips / currencyconvert

positionsize=((risk*accountBalance*convert))/(potentialLoss)
maxpositionsize=((account1.details.marginAvailable/account1.details.marginRate)/convert)/(potentialLoss)  
          
positionsize=np.minimum(positionsize,maxpositionsize)
positionsize=int(positionsize)
#    kwargs = {}
kwargs1 = {}
kwargs1["price"] = stoploss
#    
#    #self.parsed_args["stopLossOnFill"] = \
#    #kwargs["id"] = 123456
kwargs["instrument"]=pair
kwargs["units"]=sign*positionsize
kwargs["type"]="MARKET"
kwargs["stopLossOnFill"]=v20.transaction.StopLossDetails(**kwargs1)
kwargs1["price"] = takeProfit
kwargs["takeProfitOnFill"]=v20.transaction.TakeProfitDetails(**kwargs1)
    
#    
#    #marketOrderArgs.parse_arguments(args)
#    
order=api.order.market(account_id,**kwargs)
#    #order=api.order.MarketOrder(order)
#    #print("Response: {} ({})".format(order.status, order.reason))
#    #print("")re
#    #Check if trade is in same direction as current trade
#    
#    
#    #Prematurely close previous trade if signal is in opposite direction?
#    
#    
#    #Load panda
#    
#    
#    #Remove any old trades from Panda database (i.e. if stop has been triggered)
#    
#    
#    #Calculate spread
#    
#    
#    #Calculate volitility
#    
#    
#    #Calculate initial stoploss
#    
#    
#    #Place trade
#    
#    
#    #record trade to Panda along with target value at which new stop will be implemented
#    #i.e. when When profit reaches x2 the spread, 
#    #set stop loss to 1.5x the spread, if reaches x3 the spread, set trialling stop loss to x1 spread distance
#    
#    
#    #save panda to file
#        
#    return