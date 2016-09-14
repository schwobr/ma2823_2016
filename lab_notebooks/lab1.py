# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 11:43:21 2016

@author: tamernak
"""

import numpy as np
import scipy as sp

figure(1)
X=np.random.normal(size=(5000,))
h=plt.hist(X,bins=50,color='orange',histtype='stepfilled')
figure(2)
x=np.linspace(1,12,100)
y=x[:,np.newaxis]
y=y*np.cos(y)
image=y*np.sin(x)
plt.imshow(image,cmap=plt.cm.prism)
figure(3)
contours=plt.contour(image, cmap=plt.cm.prism)
plt.clabel(contours,inline=1,fontsize=10)