# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 09:03:41 2017

@author: YuanJing
"""

import bpadaboost
import numpy as np
from sklearn import svm
from sklearn.ensemble import  AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import  BaggingClassifier

dataArr,labelArr=bpadaboost.loadDataSet('eigen02.txt')
dataArr2,labelArr2=bpadaboost.loadDataSet('eigen02.txt')
dataArr.extend(dataArr2)
labelArr.extend(labelArr2)
#data01=bpadaboost.preprocess(dataArr)
#dataArr=bpadaboost.preprocess1(data01)

       
data=np.array(dataArr)
label=np.array(labelArr)

for i in range(len(label)):
    if label[i]==2:
        label[i]=-1;#adaboost只能区分-1和1的标签


#==============================================================================
# classifierArray=bpadaboost.bpadaboostTrain(data,label,1)
# 
# finalvalue,test_test=bpadaboost.adaClassify(data,classifierArray)
# 
# evaluate_train,fp=bpadaboost.evaluatemodel(label,finalvalue)
# evaluate_train=np.array(evaluate_train)
#==============================================================================


#==============================================================================
# clf=svm.SVC(C=2,kernel='rbf',gamma='auto',
#                 shrinking=True,probability=False,tol=0.001,cache_size=200,
#                 class_weight='balanced',verbose=False,max_iter=-1,
#                 decision_function_shape='ovr',random_state=None)
# clf.fit(data,label)
# label_pre=clf.predict(data)
# evaluate_train,fp=bpadaboost.evaluatemodel(label,label_pre)
# evaluate_train=np.array(evaluate_train)
#==============================================================================
clf=svm.SVC(C=2,kernel='linear',gamma='auto',shrinking=True,probability=False,
             tol=0.001,cache_size=200,class_weight='balanced',verbose=False,
             max_iter=-1,decision_function_shape='ovr',random_state=None)


clfadaboost=AdaBoostClassifier(clf,n_estimators=5,algorithm='SAMME')
clfadaboost.fit(data,label,sample_weight=None)
label_pre_ada=clfadaboost.predict(data)
evaluate_train_ada,fp_ada=bpadaboost.evaluatemodel(label,label_pre_ada)
evaluate_train_ada=np.array(evaluate_train_ada)


clfbagging=BaggingClassifier(clf,n_estimators=5)
clfbagging.fit(data,label,sample_weight=None)
label_pre_bag=clfbagging.predict(data)
evaluate_train_bag,fp_bag=bpadaboost.evaluatemodel(label,label_pre_bag)
evaluate_train_bag=np.array(evaluate_train_bag)
