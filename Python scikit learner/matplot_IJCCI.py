# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 23:59:55 2018

@author: omi
"""

import matplotlib.pyplot as plt

#classifier = ['OneR','NB Classifier', 'Bagging','C4.5','CART','Random Forest']
#accuracy = [73.69, 73.31, 90.30, 90.89, 91.33, 93.16]
classifier = ['OneR','NB Classifier', 'Bagging','C4.5','CART','Random Forest']
accuracy = [51.29, 59.94, 60.59, 65.66, 52.56, 62.84]
plt.xlabel('Classifiers')
plt.ylabel('Accuracy')
#plt.title('Classiﬁers accuracy on binary-class hand movement dataset')
plt.title('Classiﬁers accuracy on three-class hand movement dataset')
plt.plot(classifier, accuracy)
plt.show()