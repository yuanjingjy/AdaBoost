# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 08:10:52 2017

@author: YuanJing
"""
from numpy import *
from sklearn import preprocessing 

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) #get number of fields 
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
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
    for i in range(shape(dataset)[1]):#column number
        dataset[:,i]=2*dataset[:,i]-1
    return dataset

#definate the classifier first
def weakann(trainin,trainout,D):
    from sklearn.neural_network import MLPClassifier
    minerror=0;
    m,n=shape(trainin);bestann={};
    clf=MLPClassifier(hidden_layer_sizes=(90,), activation='tanh', 
                      shuffle=True,solver='sgd',alpha=1e-6,batch_size=5,
                      learning_rate='adaptive')
#==============================================================================
#     clf=MLPClassifier(solver='lbfgs',alpha=1e-5,
#                       hidden_layer_sizes=(90,),random_state=1)
#==============================================================================
    clf.fit(trainin,trainout)
    label_pre=clf.predict(trainin)
    errorArr=mat(ones((m,1)))
    errorArr[label_pre==trainout]=0
    minerror=D.T*errorArr
    bestann['clf'] = clf
    label_pre=mat(label_pre)
    return bestann,minerror,label_pre

def weaksvm(trainin,trainout,D):
    from sklearn import svm
    minerror=0;
    m,n=shape(trainin);bestsvm={};
#    clf=svm.LinearSVC()
    clf=svm.SVC(C=2,kernel='rbf',gamma='auto',
                shrinking=True,probability=False,tol=0.001,cache_size=200,
                class_weight='balanced',verbose=False,max_iter=-1,
                decision_function_shape='ovr',random_state=None)
    clf.fit(trainin,trainout)
    label_pre=clf.predict(trainin)
    errorArr=mat(ones((m,1)))
    errorArr[label_pre==trainout]=0
    minerror=D.T*errorArr
    bestsvm['clf'] = clf
    label_pre=mat(label_pre)
    return bestsvm,minerror,label_pre

def bpadaboostTrain(dataMat,labelMat,numIt):
    weakclassArr=[];m=shape(dataMat)[0]
    D=mat(ones((m,1))/m);aggClassEst=mat(zeros((m,1)))
    for i in range(numIt):
        #call the weak classifier
        #bestann,minerror,label_pre=weakann(dataMat,labelMat,D)
        bestann,minerror,label_pre=weakann(dataMat,labelMat,D)
        label_pre=label_pre.T
        print("D:",D.T)
        #calculate the weight for current weakclassifier,and update the 
        #final classifiers
        alpha=float(0.5*log((1.0-minerror)/max(minerror,1e-16)))
        bestann['alpha']=alpha
        weakclassArr.append(bestann)
        #print("label_predict:",label_pre.T)
        #update D 
        expon=multiply(-1*alpha*mat(labelMat).T,label_pre)
        D=multiply(D,exp(expon))
        D=D/D.sum()
        #update the value of final predict 
        aggClassEst += alpha*label_pre
        #print("aggClassEst:",aggClassEst.T)
        #calculate the aggregate evaluate error
        aggErrors=multiply(sign(aggClassEst)!=mat(labelMat).T,ones((m,1)))
        errorRate=aggErrors.sum()/m
        if errorRate==0.0:break
    return weakclassArr

def baggingTrain(trainin,trainout):
    trainin=array(trainin);trainout=array(trainout).T
    from sklearn.neural_network import MLPClassifier
    m=shape(trainin)[0];  weakclassArr=[];weakclass={}
    weaknum=10
    for i in range(weaknum):
        weakin=[];weakout=[]
        for j in range(m):
            randindex=random.randint(0,m)
            randsample_in=trainin[randindex].T
            randsample_out=trainout[randindex]
            weakin.append(randsample_in)
            weakout.append(randsample_out)
        clf=MLPClassifier(hidden_layer_sizes=(90,), activation='tanh', 
                          shuffle=True,solver='sgd',alpha=1e-6,batch_size=5,
                          learning_rate='adaptive')
        clf.fit(weakin,weakout)
        weakclass['clf'] = clf
        weakclassArr.append(weakclass)
    return weakclassArr
            
def adaClassify(data2classify,classifierArr):
    test_classEst=[];
    dataMatrix=mat(data2classify)
    m=shape(dataMatrix)[0]
    aggClassEst=mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst=classifierArr[i]['clf'].predict(dataMatrix)
        tmp=classifierArr[i]['alpha']
        classEst=mat(classEst).T
        test_classEst.append(classEst)
        aggClassEst += tmp*classEst
        print(aggClassEst)
    test_classEst=array(test_classEst)
    return sign(aggClassEst),test_classEst

def evaluatemodel(y_true,y_predict):
    from sklearn.metrics import confusion_matrix  
    tn, fp, fn, tp =confusion_matrix(y_true,y_predict).ravel();
    TPR=tp/(tp+fn);
    SPC=tn/(tn+fp);
    PPV=tp/(tp+fp);
    NPV=tn/(tn+fn);
    ACC=(tp+tn)/(tn+fp+fn+tp);
    return [[TPR,SPC,PPV,NPV,ACC]],[[tn,fp,fn,tp]]
    
    