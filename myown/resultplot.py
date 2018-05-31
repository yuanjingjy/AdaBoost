#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/21 8:03
# @Author  : YuanJing
# @File    : resultplot.py
"""
Description:
    1.对于ANN、AdaBoost、LR、SVM四种算法逐步增加特征值后BER的平均值及标准差，
       绘制变化过程曲线，数据文件在selectresult文件夹中
    2.找到每种算法BERmean的最小值，然后找到第一个落在最小值加减标准差范围内的索引值
        注：这里的索引值的就是实际的个数，是从1 开始的
Output message：
    stdvalue:BER的标准差
    meanvalue：BER的平均值
    minindex：BER最小值索引（从1 开始的）
    minvalue：BER最小值
    up：BER最小值位置的上限（minvalue+对应的标准差）
    down：BER最小值位置的下限
    a：up、down范围内的最小特征值数目
    tmp：a对应的BER值
"""
import  pandas as pd
import  matplotlib.pyplot as plt
import  numpy as np

sortFS=pd.read_csv('FSsort.csv')
names=sortFS['Features']#排序后特征值名称
stdresult=pd.read_csv('selectresult\LRfit.csv',names=['index','std'])
meanresult=pd.read_csv('selectresult\LRmean.csv',names=['index','mean'])

stdvalue=stdresult[1:81]['std']
meanvalue=meanresult[1:81]['mean']#提取有效数值，第一行和第一列是编号

std_up=meanvalue+stdvalue#平均值加标准差
std_down=meanvalue-stdvalue#平均值减标准差

minindex=np.argmin(meanvalue)

minvalue=meanvalue[minindex]
up=std_up[minindex]
down=std_down[minindex]
a=(meanvalue[(meanvalue<up)&(meanvalue>down)].index)[0]
tmp=meanvalue[a]


#创建画布，开始绘图
plt.figure(1)
plt.xlim(1,80)
plt.ylim(0,0.5)
x=np.linspace(1,80,80)
line_mean = plt.plot(x,meanvalue, 'r-', label='BER_mean')
line_down=plt.plot(x,std_down ,'b:', label='BER_down')
line_up=plt.plot(x,std_up, 'b:', label='BER_up')
plt.fill_between(x,std_up,std_down,color='gray',alpha=0.25)
line_h=plt.hlines(up,1,80,'r',alpha=0.25)
plt.plot(minindex,minvalue,'ro')
plt.plot(a,tmp,'r^')
plt.xticks([0,10,20,30,40,50,60,70,80,a,minindex])
line_v=plt.vlines(minindex,0,minvalue,'r',alpha=0.25)
line_v1=plt.vlines(a,0,tmp,'r',alpha=0.25)
plt.legend(loc='upper right')
plt.title('BER for LR')
plt.xlabel("Number of features")
plt.ylabel("BER")
plt.show()



print("test")
