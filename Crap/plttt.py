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
ddd=y_test-training_y_pred1[0]

plt.plot(training_y_pred[0,500:600,0])
plt.plot(training_y_pred[0,500:600,1])
#plt.plot(history1[:,0])
plt.plot(TS[500:600,0])
#plt.plot(TS[500:600,1])
#plt.plot(TS[500:600,2])
#plt.plot(TS[500:600,3])
#plt.plot(TS[500:600,4])
#plt.plot(TS[500:600,5])
plt.plot(end[2000:2200])
#plt.plot(target1[500:600,1])
#plt.plot(TS[500:600,7])
#ddd=training_y_pred-training_y_pred1
#plt.plot(tags4)
plt.show()
