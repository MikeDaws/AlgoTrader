# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 17:06:40 2018

@author: Mikw
"""

#Evaluate 
def tradeModel(predictions, Data, threshold, spread, stoploss,n):
    profit=0
    for ii in reversed(range(2,100)):
    
        if predictions[-ii,0]>threshold and predictions[-ii,1]<predictions[-ii,0]:
            if Data[-ii+1,1]-Data[-ii,3]>n*spread and Data[-ii,3]-Data[-ii+1,2]<stoploss*spread :
                profit=profit+(Data[-ii+1,1]-Data[-ii,3])-0.5*spread
            else:
                profit=profit-stoploss*spread
            
        elif predictions[-ii,1]>threshold and predictions[-ii,0]<predictions[-ii,1]:   
            if Data[-ii,3]-Data[-ii+1,2]>n*spread and Data[-ii+1,1]-Data[-ii,3]<stoploss*spread :
                profit=profit+(Data[-ii,3]-Data[-ii+1,2])-0.5*spread
            else:
                profit=profit-stoploss*spread
                
    return profit