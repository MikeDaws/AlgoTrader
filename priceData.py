
    #!/usr/bin/env python
    
import argparse
import common.config
import common.args
from datetime import datetime#
import numpy as np
import array
from datetime import datetime, timedelta
import time
import pandas as pd
    

#class priceData(self):

#    def getFullpast(a):
    
t = time.time()
    
parser = argparse.ArgumentParser()
common.config.add_argument(parser)
args = parser.parse_args()
account_id = args.config.active_account    
api = args.config.create_context()      

a="EUR_USD"

storeCandles=[]
A1=datetime.now()
store=pd.DataFrame()
tempseries=[]
for ii in range(0,10):
    kwargs = {}
    kwargs["granularity"] = "S5"
    kwargs["count"] = 5000
  
#        kwargs["toTime"] = storeCandles[ii-1][0].time
#        time.sleep(0.5)    
    while True:
        try:
            response = api.instrument.candles(a, **kwargs)        
            candles = response.get("candles", 200)
#                tempseries=pd.Series(vars(candles[0].mid))
#                df = pd.DataFrame(list(vars(candles[0].mid).items()), columns=['Date', 'DateValue','Date1', 'DateValue1'])
            for ii in range(0,len(candles)):
                if tempseries==[]:
                    tempseries=pd.DataFrame(vars(candles[ii].mid),index=[ii,])
                else:
                    tempseries=pd.DataFrame.append(vars(candles[ii].mid),index=[ii,])
    #                temp=pd.DataFrame(tempseries)
#            (store.append(temp))
            break
        except:
            time.sleep(2)
#        time.sleep(0.5)
#    storeCandles.append(candles)
#storeCandles=np.array(storeCandles)
#storeCandles=storeCandles.reshape(1,-1)
#storeCandles=np.squeeze(storeCandles)
elapsed = time.time() - t
#A=[]
#for ii in range(0,len(storeCandles)-1):
#    if storeCandles[ii].complete==True:
#        A.append(storeCandles[ii].mid.o)
#        A.append(storeCandles[ii].mid.h)
#        A.append(storeCandles[ii].mid.l)
#        A.append(storeCandles[ii].mid.c)
#        A.append(storeCandles[ii].volume)
#        
#        tempstore=datetime.strptime(storeCandles[ii].time,"%Y-%m-%dT%H:%M:%S.000000000Z")
#        A.append(tempstore.hour*60+tempstore.minute)
#
#A=np.array(A)    
#dataStore=A.reshape(-1,6)

#return dataStore