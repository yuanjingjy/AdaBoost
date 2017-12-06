# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 08:25:30 2017

@author: John
"""

import numpy as np
from sklearn import preprocessing  

def loadDataSet(filename):
    numFeat=len(open(filename).readline().split('\t'))
    dataMat=[];labelMat=[]
    fr=open(filename)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def preprocess(dataset):#将特征值规范化到[0,1]之间
    min_max_scaler=preprocessing.MinMaxScaler()
    X_train01=min_max_scaler.fit_transform(dataset)
    return X_train01

def preprocess1(dataset):#normalize the data between -1 and 1
    for i in range(np.shape(dataset)[1]):#column number
        dataset[:,i]=2*dataset[:,i]-1
    return dataset
        
def ANNClassifier(trainin,trainout,testin):
    from sklearn.neural_network import MLPClassifier#import the classifier
#==============================================================================
#     clf=MLPClassifier(hidden_layer_sizes=(90,), activation='tanh', 
#                       shuffle=True,solver='sgd',alpha=1e-6,batch_size=5,
#                       learning_rate='adaptive')
#==============================================================================
    clf=MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(90,1),
                      random_state=1)#another classifier with the solver='lbfgs'
    clf.fit(trainin,trainout)#train the classifier
    train_predict=clf.predict(trainin)#test the model with trainset
    test_predict=clf.predict(testin)#test the model with testset
    proba_train=clf.predict_proba(trainin)
    return train_predict, test_predict,proba_train

def evaluatemodel(y_true,y_predict):
    from sklearn.metrics import confusion_matrix  
    tn, fp, fn, tp =confusion_matrix(y_true,y_predict).ravel();
    TPR=tp/(tp+fn);
    SPC=tn/(fn+tn);
    PPV=tp/(tp+fp);
    NPV=tn/(tn+fn);
    ACC=(tp+tn)/(tn+fp+fn+tp);
    return [[TPR,SPC,PPV,NPV,ACC]],[[tn,fp,fn,tp]]
    