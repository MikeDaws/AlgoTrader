# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 08:30:42 2017

@author: Mikw
"""

import numpy
import talib
from talib import MA_Type

close = numpy.random.random(100)



output = talib.MOM(close, timeperiod=5)
