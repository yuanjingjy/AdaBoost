# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 16:57:41 2017

@author: John
"""

import unbalancecv
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

dataMat,labelMat=unbalancecv.loadDataSet('testdata.txt')
dataMat=np.array(dataMat)
labelMat=np.array(labelMat)

skf=StratifiedShuffleSplit(n_splits=10)
for train,test in skf.split(dataMat,labelMat):
    #print("%s %s" % (train,test))
    train_in=dataMat[train]
    test_in=dataMat[test]
    train_out=labelMat[train]
    test_out=labelMat[test]
    [num_pos_train,num_neg_train]=unbalancecv.numofclass(train_out)
    [num_pos_test,num_neg_test]=unbalancecv.numofclass(test_out)
    
    [train_in_bal,train_out_bal]=unbalancecv.oversampling(train_in,train_out)
    [num_pos_train1,num_neg_train1]=unbalancecv.numofclass(train_out_bal)