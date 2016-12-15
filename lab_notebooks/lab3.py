# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 18:19:08 2016

@author: root
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing

x = np.loadtxt('data/small_Endometrium_Uterus.csv',  delimiter=',', skiprows=1, usecols=range(1, 3001))
y = np.loadtxt('data/small_Endometrium_Uterus.csv', delimiter=',', skiprows=1, usecols=[3001], converters={3001: lambda s: 0 if s=='Endometrium' else 1}, dtype='int')
folds=cross_validation.StratifiedKFold(y,10,shuffle=True)
print folds

def cross_validate(design_matrix, labels, classifier, cv_folds):
    pred=np.zeros(labels.shape)
    for tr,te in cv_folds:
        n_samplestr=len(tr)
        n_sampleste=len(te)
        n_features=design_matrix.shape[1]
        
        xte=np.zeros((n_sampleste, n_features))
        xtr=np.zeros((n_samplestr,n_features))
        ytr=np.zeros(n_samplestr)
        
        for i in range (n_samplestr):
            xtr[i,:]=design_matrix[tr[i],:]
            ytr[i]=labels[tr[i]]
        classifier.fit(xtr,ytr)

        for i in range(n_sampleste):
            xte[i,:]=x[te[i],:]          
        predte=classifier.predict_proba(xte)
        
        for i in range (n_sampleste):
            pred[te[i]]=predte[i,1]    
    return pred
    
clf=linear_model.LogisticRegression(C=1e6)
ypred_logreg=cross_validate(x,y,clf,folds)

fpr_logreg, tpr_logreg, thresholds = metrics.roc_curve(y,ypred_logreg, pos_label=1)
print ("Accuracy : %.3f" % metrics.accuracy_score(y,np.where(ypred_logreg>0.5,1,0)))
auc_logreg = metrics.auc(fpr_logreg, tpr_logreg)

plt.figure(1)
plt.plot(fpr_logreg, tpr_logreg, '-', color='blue')
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ROC curve: Logistic regression', fontsize=16)
plt.legend(loc="lower right")

x_scaled = preprocessing.scale(x)
idx_1=0
fig = plt.figure(figsize=(12,8))

fig.add_subplot(221)
plt.hist(x[:, idx_1], bins=30, color='blue')
plt.title('Feature %d (not scaled)' % idx_1, fontsize = 16)

fig.add_subplot(223) # 2 x 2 grid, 2nd subplot
plt.hist(x_scaled[:, idx_1], bins=30, color='orange')
plt.title('Feature %d (scaled)' % idx_1, fontsize=16)

"""idx_2 = 1 # second feature

fig.add_subplot(223) # 2 x 2 grid, 3rd subplot
plt.hist(x[:, idx_2], bins=30, color='blue')
plt.title('Feature %d (not scaled)' % idx_2, fontsize=16)

fig.add_subplot(224) # 2 x 2 grid, 4th subplotplt.hist(x_scaled[:, idx_2], bins=30, color='orange')
plt.title('Feature %d (scaled)' % idx_2, fontsize=16)"""

plt.tight_layout()

def cross_validation_with_scaling(design_matrix, labels, classifier, cv_folds):
    scaler=preprocessing.StandardScaler()
    pred=np.zeros(labels.shape)
    for tr,te in cv_folds:
        n_samplestr=len(tr)
        n_sampleste=len(te)
        n_features=design_matrix.shape[1]
        
        xte=np.zeros((n_sampleste, n_features))
        xtr=np.zeros((n_samplestr,n_features))
        ytr=np.zeros(n_samplestr)
        
        for i in range (n_samplestr):
            xtr[i,:]=design_matrix[tr[i],:]
            ytr[i]=labels[tr[i]]
        xtr=scaler.fit_transform(xtr)
        classifier.fit(xtr,ytr)

        for i in range(n_sampleste):
            xte[i,:]=x[te[i],:]          

        xte=scaler.transform(xte)
        predte=classifier.predict_proba(xte)
        
        for i in range (n_sampleste):
            pred[te[i]]=predte[i,1]    
    return pred    

ypred_logreg_scaled=cross_validation_with_scaling(x,y,clf,folds)
fpr_logreg_scaled, tpr_logreg_scaled, thresholds_scaled = metrics.roc_curve(y,ypred_logreg_scaled, pos_label=1)
print ("Accuracy : %.3f" % metrics.accuracy_score(y,np.where(ypred_logreg_scaled>0.5,1,0)))
auc_logreg = metrics.auc(fpr_logreg, tpr_logreg)

plt.figure(1)
plt.plot(fpr_logreg_scaled, tpr_logreg_scaled, '-', color='orange')
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ROC curve: Logistic regression', fontsize=16)
plt.legend(loc="lower right")    