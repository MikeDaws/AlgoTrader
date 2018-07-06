import argparse
import common.config
import common.args
from datetime import datetime#
import numpy as np
import array
from datetime import datetime, timedelta
import time
import pandas as pd
import h5py    
import PyTables

class priceClass:
    
    currency="EUR_USD"
    timeFrame="S5"
    
    def _init_(self,**kwargs):
        if "currency" in kwargs:
            self.currency = kwargs["currency"]

    def saveData(self):
        store = pd.HDFStore(self.currency+"_"+'store.h5')
        store['timeseries']=self.timeseries
        store.close
        
        
    def loadData(self):
        store = pd.HDFStore(self.currency+"_"+'store.h5')
        self.timeseries = store['timeseries']
        store.close
        
        
    def changeTimeframe(self,**kwargs):
        if "timeframe" in kwargs:
            timeframe=kwargs["timeframe"]
            try:
                self.newTimeseries=self.timeseries.resample(timeframe).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})
            except:
                print("timeframe not recongised/found")
                
    def updateData(self):

        
        
        
    def getPartialhistory(self,n):
        a=self.currency
        b="S5"
        #n=1
#        t=time.time()s
        
        parser = argparse.ArgumentParser()
        common.config.add_argument(parser)
        args = parser.parse_args()
        api = args.config.create_context()
#        storeCandles=[]
#        A1=datetime.now()
        storeDict=[]
        candles=[]
        while len(storeDict)<n
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
        
        candles=None
        tempseries=pd.DataFrame(storeDict)
        storeDict=None
        tempseries=tempseries.astype({"time":datetime})
        tempseries=tempseries.set_index('time')
#        elapsed = time.time() - t
        tempseries=tempseries.sort_index()
        self.timeseries=tempseries
        
        
    def getFullhistory(self):
        a=self.currency
        b="S5"
        #n=1
#        t=time.time()s
        
        parser = argparse.ArgumentParser()
        common.config.add_argument(parser)
        args = parser.parse_args()
        api = args.config.create_context()
#        storeCandles=[]
#        A1=datetime.now()
        storeDict=[]
        candles=[]
        for ii in range(0,10000):
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
        
        candles=None
        tempseries=pd.DataFrame(storeDict)
        storeDict=None
        tempseries=tempseries.astype({"time":datetime})
        tempseries=tempseries.set_index('time')
#        elapsed = time.time() - t
        tempseries=tempseries.sort_index()
        self.timeseries=tempseries