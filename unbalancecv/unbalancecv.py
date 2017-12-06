# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 15:27:51 2017

@author: YuanJing
"""
import numpy as np

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


def oversampling(dataset,labelset):
    pos_num=0;neg_num=0
    pos_index=[];neg_index=[]
    out_data=[]
    out_label=[]
    for i in range(len(labelset)):#find the pos and neg index and number
        if labelset[i]==1:
            pos_num+=1
            pos_index.append(i)
            continue
        neg_num+=1
        neg_index.append(i)
        
    data_pos=dataset[pos_index]#dataset of positive class
    label_pos=labelset[pos_index]#labelset of positive class
    
    data_neg=dataset[neg_index]#dataset of negtive calss
    label_neg=labelset[neg_index]#labelset of negtive class
    out_data.extend(data_neg)#append the negtive data to final data
    out_label.extend(label_neg)#append the negtive label to final label
    for i in range(neg_num):#oversampling
        randindex=np.random.randint(0,pos_num)
        rand_pos_data=data_pos[randindex]
        rand_pos_label=label_pos[randindex]
        out_data.append(rand_pos_data)
        out_label.append(rand_pos_label)
    out_data=np.array(out_data)
    out_label=np.array(out_label)
    return out_data,out_label

def numofclass(labelmat):
    num_pos=0;num_neg=0
    for i in range(len(labelmat)):
        if labelmat[i]==1:
            num_pos+=1
            continue
        num_neg+=1
    return num_pos,num_neg
    
