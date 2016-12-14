# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 13:17:31 2016

@author: tamernak
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn import cross_validation

x=np.loadtxt('data/small_Endometrium_Uterus.csv', delimiter=',',skiprows=1,usecols=range(1,3001))
y=np.loadtxt('data/small_Endometrium_Uterus.csv',delimiter=',',skiprows=1, usecols=[3001],dtype='string')

y=np.where(y=='Endometrium',0,1)
print y

plt.figure(1)
idx_1=0
idx_2=1
plt.scatter(x[y==0,idx_1],x[y==0,idx_2],color='blue',label='Endometrium')
plt.scatter(x[y==1,idx_1],x[y==1,idx_2], color='orange',label='Uterus')
plt.legend(scatterpoints=1)
plt.xlabel('Gene %d'% idx_1, fontsize=14)
plt.ylabel('Gene %d' % idx_2, fontsize=14)

gnb=GaussianNB()
gnb.fit(x,y)
y_prob=gnb.predict_proba(x)
print y_prob.shape
y_pred=gnb.predict(x)
print("Number of mislabeled points out of a total %d points : %d" % (x.shape[0], (y != y_pred).sum()))
print("Accuracy : %.3f" % metrics.accuracy_score(y,y_pred))

fpr, tpr, thresholds=metrics.roc_curve(y,y_prob[:,1],pos_label=1)
auc = metrics.auc(fpr, tpr)

plt.figure(2)
plt.plot(fpr, tpr, '-', color= 'orange', label='AUC = %0.3f'%auc)
plt.xlabel('FPR', fontsize=16)
plt.ylabel('TPR', fontsize=16)
plt.title('ROC curbe : Gaussian Naive Bayes', fontsize = 16)

folds=cross_validation.StratifiedKFold(y,10,shuffle=True)

y_prob_cv=np.zeros(x.shape[0])
for ix, (tr, te) in enumerate(folds):
    n=len(tr)
    xtr=np.zeros((n,x.shape[1]))
    ytr=np.zeros(n)
    for i in range(n):
        xtr[i,:]=x[tr[i],:]
        ytr[i]=y[tr[i]]
    gnbx=GaussianNB()
    gnbx.fit(xtr,ytr)
    n=len(te)
    xte=np.zeros((n,x.shape[1]))
    for i in range(n):
        xte[i,:]=x[te[i],:]
    y_probte=gnbx.predict_proba(xte)
    for i in range(n):
        y_prob_cv[te[i]]=y_probte[i,1]
     
print("Cross-validated Accuracy test : %.3f" % metrics.accuracy_score(y,np.where(y_prob_cv>0.5,1,0)))    
fprx, tprx, thresholdsx=metrics.roc_curve(y,y_prob_cv,pos_label=1)
aucx = metrics.auc(fprx, tprx)
print aucx

plt.figure(2)
plt.plot(fprx, tprx, '-', color= 'blue', label='AUC cross-val = %0.3f'%aucx)
plt.legend(loc="lower right")

cv_aucs = cross_validation.cross_val_score(gnb,x,y,cv=folds, scoring='roc_auc')  
print "Cross-validated accuracy : %.3f" %\
metrics.accuracy_score(y, cross_validation.cross_val_predict(gnb, x, y, cv=folds)) 
print np.mean(cv_aucs)
