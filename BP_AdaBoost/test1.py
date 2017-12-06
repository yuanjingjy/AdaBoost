# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 15:36:09 2017

@author: YuanJing
"""

import bpadaboost

dataArr,labelArr=bpadaboost.loadDataSet('eigen01.txt')
testclass=bpadaboost.baggingTrain(dataArr,labelArr)