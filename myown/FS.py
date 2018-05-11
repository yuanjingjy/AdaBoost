#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/8 8:36
# @Author  : YuanJing
# @File    : FS.py

import global_new  as gl
import numpy as np
import sklearn.feature_selection as sfs

datamat=gl.dataMat
labelmat=gl.labelMat

#------------------calculate the Correlation criterion---------------------#
mean_label=np.mean(labelmat)#average of label
mean_feature=np.mean(datamat,axis=0)#average of each feature
[n_sample,n_feature]=np.shape(datamat)

corup=[]
cordown=[]
label_series=labelmat-mean_label
for j in range(n_feature):
   tmp_up=sum((datamat[:,j]-mean_feature[j])*label_series)
   corup.append(tmp_up)

    #计算相关系数公式的分母
   down_feature=np.square(datamat[:,j]-mean_feature[j])
   down_label=np.square(label_series)
   tmp_down=np.sqrt(sum(down_feature)*sum(down_label))
   cordown.append(tmp_down)

corvalue=np.array(corup)/np.array(cordown)
corvalue=np.abs(corvalue)


#------------calculate the Fisher criterion--------------#
df=np.column_stack((datamat,labelmat))#特征和标签合并

positive_set=df[df[:,80]==1]
negtive_set=df[df[:,80]==0]
positive_feaure=positive_set[:,0:80]#正类的特征
negtive_feature=negtive_set[:,0:80]#负类的特征
[sample_pos,feature_pos]=np.shape(positive_feaure)
[sample_neg,feature_neg]=np.shape(negtive_feature)

mean_pos=np.mean(positive_feaure,axis=0)#正类中，各特征的平均值
mean_neg=np.mean(negtive_feature,axis=0)#负类中，各样本的平均值
std_pos=np.std(positive_feaure,ddof=1,axis=0)#正类中各特征值的标准差
std_neg=np.std(negtive_feature,ddof=1,axis=0)#负类中各特征值的标准差
F_up=np.square(mean_pos-mean_feature)+np.square(mean_neg-mean_feature)
F_down=np.square(std_pos)+np.square(std_neg)
F_score=F_up/F_down

#------------calculate the mRMR criterion--------------#
from skfeature.function.similarity_based import fisher_score
from skfeature.function.information_theoretical_based import MRMR
score = fisher_score.fisher_score(datamat, labelmat)
idx=fisher_score.feature_ranking(score)
score1=MRMR.mrmr(datamat,labelmat)

print("test")
