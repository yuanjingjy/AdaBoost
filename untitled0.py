# -*- coding: utf-8 -*-
"""
Created on Tue May 29 09:56:40 2018

@author: YJ
"""
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix  
#    from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
#    from sklearn.metrics import precision_score
#    from sklearn.metrics import recall_score

data=pd.read_csv('MIV30.csv')
data=data.fillna(1)
train_data=data.iloc[:,0:10]
#train_data=round(train_data,2)
train_label=np.array(data.loc[:,'class_label'])
#dtrain=xgb.DMatrix(train_data,label=train_label)
clf=xgb.XGBClassifier(max_depth=100,learning_rate=0.001,n_estimators=100,n_jobs=-1,random_state=100,silent=False)
clf.fit(train_data,train_label)
pre=clf.predict(train_data)
pre_num=clf.predict_proba(train_data)

clf1=LogisticRegression(random_state=100)
clf1.fit(train_data,train_label)
pre1=clf1.predict(train_data)
pre_num1=clf1.predict_proba(train_data)

print("test")


tn, fp, fn, tp =confusion_matrix(train_label,pre).ravel();
TPR=tp/(tp+fn);
SPC=tn/(fp+tn);
PPV=tp/(tp+fp);
NPV=tn/(tn+fn);
acc=(tp+tn)/(tn+fp+fn+tp);
auc=roc_auc_score(train_label,pre_num[:,1])