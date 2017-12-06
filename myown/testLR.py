# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 14:44:25 2017

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 08:42:39 2017

@author: John
"""

import ann
import logRegres as LR
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold

import  pandas as  pd#python data analysis
import  sklearn.feature_selection as sfs

pandas_data=pd.read_excel('sql_eigen.xlsx',header=None)
sql_eigen=pandas_data.fillna(np.mean(pandas_data))

data =sql_eigen.iloc[:,0:85]
data.iloc[:,84][data.iloc[:,84]>200]=91
label=sql_eigen.iloc[:,86]

dataMat1=np.array(data)
labelMat=np.array(label)

data01 = ann.preprocess(dataMat1)
# dataMat = ann.preprocess1(data01)
dataMat=np.array(data01)

for i in range(len(labelMat)):
    if labelMat[i] == -1:
        labelMat[i] = 0;  # adaboost只能区分-1和1的标签

evaluate_train = []
evaluate_test = []
prenum_train = []
prenum_test  = []

clf1=MLPClassifier(hidden_layer_sizes=(90,), activation='tanh',
                      shuffle=True,solver='sgd',alpha=1e-6,batch_size=5,
                      learning_rate='adaptive')

clf=svm.SVC(C=2,kernel='linear',gamma='auto',shrinking=True,probability=True,
             tol=0.001,cache_size=200,class_weight='balanced',verbose=False,
             max_iter=-1,decision_function_shape='ovr',random_state=None)
#

dataMat_new=sfs.SelectKBest(sfs.chi2,k=50).fit_transform(dataMat,labelMat)
dataMat=dataMat_new


num_sample=np.shape(dataMat1)[0]
addones=np.ones((num_sample,1))
dataMat=np.c_[addones,dataMat]
data01=ann.preprocess(dataMat)
dataMat1=ann.preprocess1(data01)

dataMat=dataMat1;
#dataMat=dataMat1[:,0:78]
#dataMat=dataMat[:,0:44]

#==============================================================================
# for i in range(len(labelMat)):
#     if labelMat[i]==2:
#         labelMat[i]=0;#adaboost只能区分-1和1的标签
#==============================================================================

evaluate_train=[]
evaluate_test=[]
prenum_train=[]
prenum_test=[]

skf=StratifiedKFold(n_splits=10)
for train,test in skf.split(dataMat,labelMat):
#==============================================================================
# skf=StratifiedShuffleSplit(n_splits=10)
# for train,test in skf.split(dataMat,labelMat):
#==============================================================================
    print("%s %s" % (train,test))
    train_in=dataMat[train]
    test_in=dataMat[test]
    train_out=labelMat[train]
    test_out=labelMat[test]
    
    trainWeights=LR.stocGradAscent1(train_in,train_out,200)
    
    len_train=np.shape(train_in)[0]
    len_test=np.shape(test_in)[0]
    test_predict=[]
    proba_test=[]
    for i in range(len_test):
        test_predict_tmp=LR.classifyVector(test_in[i,:], trainWeights)
        test_predict.append(test_predict_tmp)
        proba_test_tmp=LR.classifyProb(test_in[i,:], trainWeights)
        proba_test.append(proba_test_tmp)
     
        
    train_predict=[]
    proba_train=[]
    for i in range(len_train):
        train_predict_tmp=LR.classifyVector(train_in[i,:], trainWeights)
        train_predict.append(train_predict_tmp)
        proba_train_tmp=LR.classifyProb(train_in[i,:], trainWeights)
        proba_train.append(proba_train_tmp)
  
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


print(evaluate_test)