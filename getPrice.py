
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
    

a="EUR_USD"
b="S5"
#n=1
t=time.time()

parser = argparse.ArgumentParser()
common.config.add_argument(parser)
args = parser.parse_args()
api = args.config.create_context()
storeCandles=[]
A1=datetime.now()
currentTime=datetime.now()
#print("%Y-%m-%dT%H:%M:%S.000000000Z",(currentTime.year, currentTime.month, currentTime.day,currentTime.hour, currentTime.minute, currentTime.second))

str(currentTime.year)+"-"+str(currentTime.month)+"-"+str(currentTime.day)+"T"+str(currentTime.hour)+":"+str(currentTime.minute)+":"+str(currentTime.second)+".000000000Z"

storeDict=[]
candles=[]
for ii in range(0,10):
        kwargs = {}
        kwargs["granularity"] = b
        kwargs["count"] = 5000
   
        if ii>0:
            kwargs["toTime"] = candles[0].time
#        time.sleep(0.5)    
        while True:
            try:
                response = api.instrument.candles(a, **kwargs)
                candles = response.get("candles", 200)
                break
            except:
             time.sleep(2)
             print("error")
    #        time.sleep(0.5)
        if candles != []:
            for jj in range(0,len(candles)):
                temp1=vars(candles[jj].mid)
                temp2={"volume":candles[jj].volume}
                temp3={"time":datetime.strptime(candles[jj].time,"%Y-%m-%dT%H:%M:%S.000000000Z")}
                dict1={**temp1, **temp2, **temp3}
                storeDict.append(dict1)

#            storeCandles.append(candles)
            print(ii)

#candles=None
tempseries=pd.DataFrame(storeDict)
storeDict=None
tempseries=tempseries.astype({"time":datetime})
tempseries=tempseries.set_index('time')
elapsed = time.time() - t
tempseries=tempseries.sort_index()