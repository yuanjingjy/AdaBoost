# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 08:42:39 2017

@author: John
"""

import ann
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier#import the classifier
from sklearn.feature_selection import  RFE#
from sklearn import svm
import  pandas as  pd#python data analysis
import  sklearn.feature_selection as sfs
import matplotlib.pyplot as plt
from sklearn import preprocessing

pandas_data=pd.read_csv('sql_eigen.csv')
sql_eigen=pandas_data.fillna(np.mean(pandas_data))

data =sql_eigen.iloc[:,0:85]
# data.iloc[:,84][data.iloc[:,84]>200]=91
data['age'][data['age']>200]=91
label=sql_eigen['class_label']

dataMat1=np.array(data)
labelMat=np.array(label)

data01 = ann.preprocess(dataMat1)
# dataMat = ann.preprocess1(data01)
dataMat=np.array(data01)

dataandlabel=pd.DataFrame(labelMat,columns=['label'])
dataandlabel.to_csv('F:/label.csv', encoding='utf-8', index=True)
for i in range(len(labelMat)):
    if labelMat[i] == -1:
        labelMat[i] = 0;  # adaboost只能区分-1和1的标签

evaluate_train = []
evaluate_test = []
prenum_train = []
prenum_test  = []
weight=[]
######################################################
#########select features with selectKBest#####################
# selectmodel=sfs.SelectKBest(sfs.mutual_info_classif)
# selectmodel.fit(dataMat,labelMat)
# selectedeigen=selectmodel.get_support()
# selectedscore=selectmodel.scores_
# selectedpvalue=selectmodel.pvalues_
# score=ann.rowscale(selectedscore)
# # pvalue=ann.rowscale(selectedpvalue)
# compareeigen=pd.DataFrame([score,selectedeigen],index=['score','YN'],columns=data.keys())
# sorteigen=compareeigen.sort_values(by='score',ascending=False,axis=1)
# sorteigen.to_csv('F:/ftest.csv', encoding='utf-8', index=True)
# datatoplot=sorteigen.iloc[0]
# plt.figure()
# datatoplot.plot(title='Score of chi2 for eigens',legend=True,stacked=True,alpha=0.7)
# plt.show()
# dataMat_new=sfs.SelectKBest(sfs.chi2,k=50).fit_transform(dataMat,labelMat)
# # indices=sfs.SelectKBest(sfs.chi2,k=50).get_support()
# dataMat=dataMat_new
######################################################

clf1=MLPClassifier(hidden_layer_sizes=(90,), activation='tanh',
                      shuffle=True,solver='sgd',alpha=1e-6,batch_size=5,
                      learning_rate='adaptive')

clf=svm.SVC(C=2,kernel='linear',gamma='auto',shrinking=True,probability=True,
             tol=0.001,cache_size=200,class_weight='balanced',verbose=False,
             max_iter=-1,decision_function_shape='ovr',random_state=None)
score =[]
for i in range(84):
    rfe = RFE(estimator=clf, n_features_to_select=i+1, step=1)
    rfe.fit(dataMat, labelMat)
    selectedeigen = rfe.support_
    selectedscore = rfe.score(dataMat, labelMat)
    score.append(selectedscore)


# selectedpvalue=selectmodel.pvalues_
# compareeigen=pd.DataFrame([selectedscore,selectedpvalue,selectedeigen],index=['score','pvalue','YN'],columns=data.keys())
# sorteigen=compareeigen.sort_values(by='score',ascending=False,axis=1)
num_features = rfe.n_features_
select_feature = rfe.support_
rank_fea = rfe.ranking_
model = rfe.estimator_

compareeigen=pd.DataFrame([rank_fea,selectedeigen],index=['rank','YN'],columns=data.keys())
sorteigen=compareeigen.sort_values(by='rank',ascending=False,axis=1)
#
# data_mat=dataMat(select_feature)
skf = StratifiedKFold(n_splits=10)
for train, test in skf.split(dataMat, labelMat):
    # ==============================================================================
    # skf=StratifiedShuffleSplit(n_splits=10)
    # for train,test in skf.split(dataMat,labelMat):
    # ==============================================================================
    print("%s %s" % (train, test))
    train_in = dataMat[train]
    test_in = dataMat[test]
    train_out = labelMat[train]
    test_out = labelMat[test]
    train_predict, test_predict, proba_train, proba_test,weights = ann.SVMClassifier(train_in, train_out, test_in)
    weight.append(weights)
    proba_train = proba_train[:, 1]
    proba_test = proba_test[:, 1]
    test1, test2 = ann.evaluatemodel(train_out, train_predict, proba_train)  # test model with trainset
    evaluate_train.extend(test1)
    prenum_train.extend(test2)

    test3, test4 = ann.evaluatemodel(test_out, test_predict, proba_test)  # test model with testset
    evaluate_test.extend(test3)
    prenum_test.extend(test4)

mean_train = np.mean(evaluate_train, axis=0)
std_train = np.std(evaluate_train, axis=0)
evaluate_train.append(mean_train)
evaluate_train.append(std_train)

mean_test = np.mean(evaluate_test, axis=0)
std_test = np.std(evaluate_test, axis=0)
evaluate_test.append(mean_test)
evaluate_test.append(std_test)

evaluate_train = np.array(evaluate_train)
evaluate_test = np.array(evaluate_test)
prenum_train = np.array(prenum_train)
prenum_test = np.array(prenum_test)

evaluate_train_mean = np.mean(evaluate_test, axis=0)
# np.array(test_important)
weight=np.array(weight)
print("view the variable")