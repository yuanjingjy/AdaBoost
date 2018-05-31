#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/29 16:39
# @Author  : YuanJing
# @File    : comparemodel.py

import  numpy as np
import  pandas as pd
import  matplotlib.pyplot as plt

adaboost=pd.read_csv('BER/BER_AdaBoost.csv')
ann=pd.read_csv('BER/BER_ANN.csv')
logistic=pd.read_csv('BER/BER_LR.csv')
svm=pd.read_csv('BER/BER_SVM.csv')
mews=pd.read_csv('BER/BER_MEWS.csv')

TPR=pd.DataFrame()
TPR['ANN']=ann['TPR']
TPR['LR']=logistic['TPR']
TPR['SVM']=svm['TPR']
TPR['AdaBoost']=adaboost['TPR']
TPR['MEWS']=mews['TPR']

SPC=pd.DataFrame()
SPC['ANN']=ann['SPC']
SPC['LR']=logistic['SPC']
SPC['SVM']=svm['SPC']
SPC['AdaBoost']=adaboost['SPC']
SPC['MEWS']=mews['SPC']

PPV=pd.DataFrame()
PPV['ANN']=ann['PPV']
PPV['LR']=logistic['PPV']
PPV['SVM']=svm['PPV']
PPV['AdaBoost']=adaboost['PPV']
PPV['MEWS']=mews['PPV']

NPV=pd.DataFrame()
NPV['ANN']=ann['NPV']
NPV['LR']=logistic['NPV']
NPV['SVM']=svm['NPV']
NPV['AdaBoost']=adaboost['NPV']
NPV['MEWS']=mews['NPV']

ACC=pd.DataFrame()
ACC['ANN']=ann['ACC']
ACC['LR']=logistic['ACC']
ACC['SVM']=svm['ACC']
ACC['AdaBoost']=adaboost['ACC']
ACC['MEWS']=mews['ACC']

AUC=pd.DataFrame()
AUC['ANN']=ann['AUC']
AUC['LR']=logistic['AUC']
AUC['SVM']=svm['AUC']
AUC['AdaBoost']=adaboost['AUC']
AUC['MEWS']=mews['AUC']

BER=pd.DataFrame()
BER['ANN']=ann['BER']
BER['LR']=logistic['BER']
BER['SVM']=svm['BER']
BER['AdaBoost']=adaboost['BER']
BER['MEWS']=mews['BER']

ACC.boxplot()
plt.show()

print('test')