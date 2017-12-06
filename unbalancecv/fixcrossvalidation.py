# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 10:37:34 2017

@author: YuanJing
"""


import adaboost
import unbalancecv
import numpy as np
from sklearn.model_selection import StratifiedKFold

train_in=[];
test_in=[];
train_out=[];
test_out=[];
test_index=[];
cross_count=1;

outdir="E:\splitdataset"

#==============================================================================
# if not os.path.exists(outdir): #if not outdir,makedir
#         os.makedirs(outdir)
#==============================================================================

dataArr,labelArr=adaboost.loadDataSet('alldata.txt');

eigen=np.array(dataArr);
label=np.array(labelArr);

skf=StratifiedKFold(n_splits=10,shuffle=False,random_state=None)
for train,test in skf.split(dataArr,labelArr):
    print("%s %s" % (train, test))
    test_index.append(test)
    train_in=eigen[train];
    test_in=eigen[test];
    train_out=label[train];
    test_out=label[test];
    [train_in_bal,train_out_bal]=unbalancecv.oversampling(train_in,train_out)
    np.savetxt(outdir + "/train_in_" + str(cross_count) + '.txt',\
                        train_in_bal,fmt="%s", delimiter='\t')
    np.savetxt(outdir + "/train_out_" + str(cross_count) + '.txt',\
                        train_out_bal,fmt="%s", delimiter='\t')
    np.savetxt(outdir + "/test_in_" + str(cross_count) + '.txt',\
                        test_in,fmt="%s", delimiter='\t')
    np.savetxt(outdir + "/test_out_" + str(cross_count) + '.txt',\
                        test_out,fmt="%s", delimiter='\t')
    cross_count+=1