# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 08:42:39 2017

@author: John
"""

import ann
import  pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold
########全部的特征值###################################3
# dataMat1,labelMat=ann.loadDataSet('final_shuffle.txt')
# #==============================================================================
# # dataArr2,labelArr2=ann.loadDataSet('final_eigen02.txt')
# # dataMat1.extend(dataArr2)
# # labelMat.extend(labelArr2)
# #==============================================================================
# labelMat=np.array(labelMat)
#
#
# data01=ann.preprocess(dataMat1)
# dataMat=ann.preprocess1(data01)
# #dataMat=np.array(data01)
# #dataMat=np.array(dataMat1)
#
# #dataMat=dataMat[:,0:44]
########全部的特征值###################################3

###################遗传算法降维后的特征值#################
dataset=pd.read_csv("LR9.csv")
dataset=np.array(dataset)
dataMat=dataset[:, 0:9]
dataMat=ann.preprocess(dataMat)
dataMat=ann.preprocess1(dataMat)
labelMat=dataset[:,9]


###################遗传算法降维后的特征值#################

# import  global_list as gl
# dataset=gl.dataSet
# dataset=np.array(dataset)
# dataMat=dataset[:,0:78]
# labelMat=dataset[:,78]

# for i in range(len(labelMat)):
#     if labelMat[i]==2:
#         labelMat[i]=-1;#adaboost只能区分-1和1的标签

evaluate_train=[]
evaluate_test=[]
prenum_train=[]
prenum_test=[]

skf = StratifiedKFold(n_splits=10)
for train, test in skf.split(dataMat, labelMat):
    print("%s %s" % (train, test))
    train_in = dataMat[train]
    test_in=dataMat[test]
    train_out=labelMat[train]
    test_out=labelMat[test]
    train_in, train_out = RandomUnderSampler().fit_sample(train_in, train_out)
    train_predict,test_predict,proba_train,proba_test=ann.ANNClassifier(train_in,train_out,test_in)
    proba_train=proba_train[:,1]
    proba_test=proba_test[:,1]
    test1,test2=ann.evaluatemodel(train_out,train_predict,proba_train)#test model with trainset
    evaluate_train.extend(test1)
    prenum_train.extend(test2)
    
    test3,test4=ann.evaluatemodel(test_out,test_predict,proba_test)#test model with testset
    evaluate_test.extend(test3)
    prenum_test.extend(test4)
    
mean_train=np.mean(evaluate_train,axis=0)
std_train=np.std(evaluate_train,axis=0)
evaluate_train.append(mean_train)
evaluate_train.append(std_train)

mean_test=np.mean(evaluate_test,axis=0)
std_test=np.std(evaluate_test,axis=0)
evaluate_test.append(mean_test)
evaluate_test.append(std_test)
    
evaluate_train=np.array(evaluate_train)
evaluate_test=np.array(evaluate_test)
prenum_train=np.array(prenum_train)
prenum_test=np.array(prenum_test)

evaluate_train_mean=np.mean(evaluate_test,axis=0)
#np.array(test_important)
print("view the variable")