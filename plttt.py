# -*- coding: utf-8 -*-
"""
Created on Tue May  1 20:01:41 2018

@author: Mikw
"""
import matplotlib
import matplotlib.pyplot as plt


tags4=[]
for i in range(0,500):
    tags4.append(0)
tags4.extend(tags2)

plt.plot(training_y_pred[0,:])
#plt.plot(tags4)
plt.show()
