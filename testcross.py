# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 16:09:36 2017

@author: YuanJing
"""
import adaboost
import numpy as np
from sklearn.model_selection import StratifiedKFold
import  pandas as pd

train_in=[];
test_in=[];
train_out=[];
test_out=[];
errnum=[];
evaluate_train=[];
evaluate_test=[];
AUCaaa=[];
fp_train=[];
fp_test=[];
AUCtest=[]

###
# ######加载全部62个特征值
# import  global_list as gl
# dataset=gl.dataSet
# dataset=np.array(dataset)
# dataArr=dataset[:,0:62]
# labelArr=dataset[:,62]

######加载遗传算法降维后的35个特征值（MIV降维后的时eigen_MIV，34个）
dataset=pd.read_csv("GA31.csv")
dataset=np.array(dataset)
dataArr=dataset[:, 0:31]
labelArr=dataset[:,31]

# ######加载MIV降维后的30个特征值
# dataset=pd.read_csv("MIV30.csv")
# dataset=np.array(dataset)
# dataArr=dataset[:, 0:30]
# labelArr=dataset[:,30]

# ######加载逻辑回归降维后的10个特征值
# dataset=pd.read_csv("LR10.csv")
# dataset=dataset.fillna(np.mean(dataset))
# dataset=np.array(dataset)
# dataArr=dataset[:, 0:10]
# # dataMat=ann.preprocess(dataMat)
# # dataMat=ann.preprocess1(dataMat)
# labelArr=dataset[:,10]

for i in range(len(labelArr)):
    if labelArr[i]==0:
        labelArr[i]=-1;#adaboost只能区分-1和1的标签

# dataArr=dataMat
label=labelArr
skf=StratifiedKFold(n_splits=10)
for train,test in skf.split(dataArr,labelArr):
    # print("%s %s" % (train, test))
    train.tolist();
    train_in=dataArr[train];
    test_in=dataArr[test];
    train_out=label[train];
    test_out=label[test];
    # train_in, train_out = RandomUnderSampler().fit_sample(train_in, train_out)
    classifierArray,aggClassEst=adaboost.adaBoostTrainDS(train_in,train_out,200);
    prediction_train,prob_train=adaboost.adaClassify(train_in,classifierArray);#测试训练集
    prediction_test,prob_test=adaboost.adaClassify(test_in,classifierArray);#测试测试集
    AUC_tmp=adaboost.plotROC(aggClassEst.T,train_out);#计算AUC
    AUCaaa.append(AUC_tmp);
    AUC_test_tmp=adaboost.plotROC(prob_test.T,test_out);#计算AUC
    AUCtest.append(AUC_test_tmp);
    tmp_train,fp_train_tmp=adaboost.evaluatemodel(train_out,prediction_train);
    #evaluate_train=np.array(evaluate_train);
    evaluate_train.extend(tmp_train);#训练集结果评估
    fp_train.extend(fp_train_tmp);
    
    tmp_test,fp_test_tmp=adaboost.evaluatemodel(test_out,prediction_test);
    evaluate_test.extend(tmp_test);
    fp_test.extend(fp_test_tmp);
    
mean_train=np.mean(evaluate_train,axis=0)
std_train=np.std(evaluate_train,axis=0)
evaluate_train.append(mean_train)
evaluate_train.append(std_train)

mean_test=np.mean(evaluate_test,axis=0)
std_test=np.std(evaluate_test,axis=0)
evaluate_test.append(mean_test)
evaluate_test.append(std_test)
    
evaluate_train=np.array(evaluate_train);
evaluate_test=np.array(evaluate_test);
fp_train=np.array(fp_train);
fp_test=np.array(fp_test);
AUC=np.mean(AUCtest)
print("test")
