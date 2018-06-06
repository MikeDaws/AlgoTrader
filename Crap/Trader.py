#Main File for doing shit
#import panda
import talib
import algo.get
import algo.getpast
import common.config
import common.args
import datetime

#Parameter list
loadPrev = True
trainOn = False
filePath = "C:/"


history=algo.getpast.getpast("EUR_USD","H1")




##Current time
#[a,b]=algo.get.get("EUR_USD")
#
#date=split('T')
#datetime.
#



#
##Function lists (until I put these somewhere else or start using a class system)
#
#
#
#def loadData(filePath):
#    #load csv here
#    
#    
#    dataPanda=1
#    return dataPanda
#
#
#
#
#
#
#def trainData(dataPanda,extra):
#    
#    
#    rnnObject = 1
#    return rnnObject
#
#
#
#
#
#
#
#
#
#
#
#
#
#
##Load in historic data into a panda
#
#if loadPrev == True:
#    loadData(filePath)
#
#
#    #Carry out some quant analysis on the data i.e. moving averages etc
#
#    extra=1
#
#
#
##Train data
#if trainOn == True:
#    trainData(dataPanda,extra)
#
#
##Retrieve latest price data
#    
#    
#    
    