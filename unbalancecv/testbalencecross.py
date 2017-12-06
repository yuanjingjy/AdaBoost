# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 16:09:36 2017

@author: John
"""
import adaboost
import unbalancecv
import numpy as np
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

dataArr,labelArr=adaboost.loadDataSet('alldata.txt');


for i in range(len(labelArr)):
    if labelArr[i]==0:
        labelArr[i]=-1;#adaboost只能区分-1和1的标签

eigen=np.array(dataArr);
label=np.array(labelArr);

skf=StratifiedKFold(n_splits=10,shuffle=False,random_state=None)
for train,test in skf.split(dataArr,labelArr):

    print("%s %s" % (train, test))
    train_in=eigen[train];
    test_in=eigen[test];
    train_out=label[train];
    test_out=label[test];
    
    [train_in_bal,train_out_bal]=unbalancecv.oversampling(train_in,train_out)
    
    classifierArray,aggClassEst=adaboost.adaBoostTrainDS(train_in_bal,train_out_bal,50) 
        
    prediction_train=adaboost.adaClassify(train_in_bal,classifierArray)#测试训练集
    prediction_test=adaboost.adaClassify(test_in,classifierArray);#测试测试集
#==============================================================================
#     AUC_tmp=adaboost.plotROC(aggClassEst.T,train_out_bal)#计算AUC
#     AUCaaa.append(AUC_tmp)
#     AUCbbb=np.array(AUCaaa)
#==============================================================================
    tmp_train,fp_train_tmp=adaboost.evaluatemodel(train_out_bal,prediction_train);
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
    