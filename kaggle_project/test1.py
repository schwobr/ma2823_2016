# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 20:56:50 2016

@author: root
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing

x = np.loadtxt('data/train.csv',  delimiter=',', skiprows=1, usecols=range(2, 14))
y = np.loadtxt('data/train.csv', delimiter=',', skiprows=1, usecols=[14])
folds=cross_validation.StratifiedKFold(y,10,shuffle=True,random_state=42)

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
        predte=classifier.predict(xte)
        
        for i in range (n_sampleste):
            pred[te[i]]=predte[i]    
    return pred    

clf=linear_model.LogisticRegression(C=1e6)

"""ypred_logreg_scaled=cross_validation_with_scaling(x,y,clf,folds)
fpr_logreg_scaled, tpr_logreg_scaled, thresholds_scaled = metrics.roc_curve(y,ypred_logreg_scaled, pos_label=1)
print ("Accuracy : %.3f" % metrics.accuracy_score(y,np.where(ypred_logreg_scaled>0.5,1,0)))
auc_logreg_scaled = metrics.auc(fpr_logreg_scaled, tpr_logreg_scaled)

plt.figure(1)
plt.plot(fpr_logreg_scaled, tpr_logreg_scaled, '-', color='orange')
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ROC curve: Logistic regression', fontsize=16)
plt.legend(loc="lower right")"""