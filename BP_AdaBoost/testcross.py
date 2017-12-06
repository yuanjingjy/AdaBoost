# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 16:09:36 2017

@author: YuanJing
"""
import bpadaboost
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
 

#reset() 

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

dataArr,labelArr=bpadaboost.loadDataSet('final_data.txt')
#==============================================================================
# dataArr2,labelArr2=bpadaboost.loadDataSet('eigen02.txt')
# dataArr.extend(dataArr2)
# labelArr.extend(labelArr2)
#==============================================================================
data01=bpadaboost.preprocess(dataArr)
dataArr=bpadaboost.preprocess1(data01)

for i in range(len(labelArr)):
    if labelArr[i]==2:
        labelArr[i]=-1;#adaboost只能区分-1和1的标签

eigen=np.array(dataArr);
label=np.array(labelArr);

#==============================================================================
# skf=StratifiedShuffleSplit(n_splits=10)
# for train,test in skf.split(dataArr,labelArr):
#==============================================================================
skf=StratifiedKFold(n_splits=10)
for train,test in skf.split(dataArr,labelArr):
    print("%s %s" % (train, test))
    train.tolist();
    train_in=eigen[train];
    test_in=eigen[test];
    train_out=label[train];
    test_out=label[test];
    classifierArray=bpadaboost.bpadaboostTrain(train_in,train_out,5);            
    prediction_train,test_train=bpadaboost.adaClassify(train_in,classifierArray);#测试训练集
    prediction_test,test_test=bpadaboost.adaClassify(test_in,classifierArray);#测试测试集
#==============================================================================
#     AUC_tmp=adaboost.plotROC(aggClassEst.T,train_out);#计算AUC
#     AUCaaa.append(AUC_tmp);
#==============================================================================
    tmp_train,fp_train_tmp=bpadaboost.evaluatemodel(train_out,prediction_train);
    #evaluate_train=np.array(evaluate_train);
    evaluate_train.extend(tmp_train);#训练集结果评估
    fp_train.extend(fp_train_tmp);
    
    tmp_test,fp_test_tmp=bpadaboost.evaluatemodel(test_out,prediction_test);
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
    