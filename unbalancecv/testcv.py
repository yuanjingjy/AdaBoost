# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 15:30:26 2017

@author: John
"""

if __name__ == '__main__':
    os.chdir("your workhome") #你的数据存放目录
    datadir = "split10_1" #切分后的文件输出目录
    splitDataSet('datasets',10,datadir)#将数据集datasets切为十个保存到datadir目录中
    #==========================================================================
    outdir = "sample_data1" #抽样的数据集存放目录
    train_all,test_all = generateDataset(datadir,outdir) #抽样后返回训练集和测试集
    print "generateDataset end and cross validation start"
    #==========================================================================
    #分类器部分
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=500) #使用随机森林分类器来训练
    clfname = "RF_1"    #==========================================================================
    curdir = "experimentdir" #工作目录
    #clf:分类器,clfname:分类器名称,curdir:当前路径,train_all:训练集,test_all:测试集
    ACC_mean, SN_mean, SP_mean = crossValidation(clf, clfname, curdir,train_all,test_all)
    print ACC_mean,SN_mean,SP_mean  #将ACC均值，SP均值，SN均值都输出到控制台