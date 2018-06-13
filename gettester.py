
    #!/usr/bin/env python
    
import argparse
import common.config
import common.args
from datetime import datetime#
import numpy as np
import array
from datetime import datetime, timedelta
import time
    
#def getFullpast(a,b):

b="M15"
a="GBP_USD"
parser = argparse.ArgumentParser()

#
# The config object is initialized by the argument parser, and contains
# the REST APID host, port, accountID, etc.
#
common.config.add_argument(parser)








parser = argparse.ArgumentParser()

common.config.add_argument(parser)
args = parser.parse_args()
account_id = args.config.active_account

api = args.config.create_context()





args = parser.parse_args()

account_id = args.config.active_account

#
# The v20 config object creates the v20.Context for us based on the
# contents of the config file.
#
api = args.config.create_context()
storeCandles=[]
A1=datetime.now()
for ii in range(0,1):
    kwargs = {}
    kwargs["granularity"] = b
    kwargs["count"] = 5000
#    datet=str(A1.year)+"-"+str(A1.month)+"-"+str(A1.day)+"T"+str(A1.hour)+":"+str(A1.minute)+":"+str(A1.second)+"."+str(A1.microsecond)
#    A1=A1-timedelta(minutes=15*4999)
#    kwargs["toTime"] = datetime.now()    
    if ii>0:
        kwargs["toTime"] = storeCandles[ii-1][0].time
    response = api.instrument.candles(a, **kwargs)
    
    candles = response.get("candles", 200)   
    storeCandles.append(candles)
storeCandles=np.array(storeCandles)
storeCandles=storeCandles.reshape(1,-1)
storeCandles=np.squeeze(storeCandles)

A=[]
for ii in range(0,len(storeCandles)-1):
    A.append(storeCandles[ii].mid.o)
    A.append(storeCandles[ii].mid.h)
    A.append(storeCandles[ii].mid.l)
    A.append(storeCandles[ii].mid.c)
    A.append(storeCandles[ii].volume)
    
    tempstore=datetime.strptime(storeCandles[ii].time,"%Y-%m-%dT%H:%M:%S.000000000Z")
    A.append(tempstore.hour*60+tempstore.minute)

A=np.array(A)    
dataStore=A.reshape(-1,6)

#    return dataStore