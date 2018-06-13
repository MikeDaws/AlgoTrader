# -*- coding: utf-8 -*-
"""
Created on Sun May 27 09:54:07 2018

@author: Mikw
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 17:38:45 2017

@author: Mikw
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 10:53:29 2017

@author: Mikw
"""
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
import argparse
import common.config
import common.args

from pandas import DataFrame
from pandas import concat
from skopt.space import Real, Integer
from skopt.utils import use_named_args

import tensorflow as tf
import shutil
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.layers as tflayers
from tensorflow.contrib.learn.python.learn import learn_runner
import tensorflow.contrib.metrics as metrics
import tensorflow.contrib.rnn as rnn
from account.account import Account
from order.args import OrderArguments
from optfunc import optFunc
#Main File for doing shit
#import panda
import talib
import algo.get
import algo.getpast
import common.config
import common.args
import datetime
import getFullpast
from skopt import gp_minimize, gbrt_minimize
from notest import notest
from getSpread import getSpread

pair="EUR_JPY"
Data=(getFullpast.getFullpast(pair,"M15",11))
np.save("./save",Data)


#layers_stacked_count=x[0]
#hidden=x[1]
#beta=x[2]
#threshold=x[3]
#spread=0.00015
#stoploss=x[4]

space  = [Real(0.1, 0.5, name='Threshold'),
          Real(0.5, 3, name='stoploss'),
          Real(1.5, 4, name='profitmulti'),
          Real(2, 4, name='trainmulti')]
#Data=np.load('./save.npy')
#hidden=50
#layers_stacked_count=3
#beta=0.0001
#notest(pair,1,Data[:-100],0.00015, hidden, layers_stacked_count, beta) 
  
#space  = [Real(0, 1, name='Threshold'),
#          Real(0, 0.001, name='stoploss')]

use_named_args(space)

res=gbrt_minimize(optFunc, space, n_calls=60, random_state=0,n_random_starts=10, verbose='True', n_jobs=1 )
