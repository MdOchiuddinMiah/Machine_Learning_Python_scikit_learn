# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 21:53:10 2018

@author: omi
"""

from sklearn import tree
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix

#eeg_data=pd.read_csv("E:\\Research\\ensemble classifier in BMI\\2_3_class_data\\2omitrain.csv")
#eeg_test_data=pd.read_csv("E:\\Research\\ensemble classifier in BMI\\2_3_class_data\\2omitest.csv")

eeg_data=pd.read_csv("E:\\Research\\ensemble classifier in BMI\\sensors data\\2class\\2F4_train.csv")
eeg_test_data=pd.read_csv("E:\\Research\\ensemble classifier in BMI\\sensors data\\2class\\2F4_test.csv")

eeg_data.head()
featute_col=['Theta','Alpha','Low_beta','High_beta','Gamma']

X_train=eeg_data[featute_col]
y_train=eeg_data['Class']

X_test=eeg_test_data[featute_col]
y_test=eeg_test_data['Class']

print('ok')

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

print('Accuracy on the test subset: {:.3f}'.format(clf.score(X_test, y_test)))

predicted=clf.predict(X_test)
print('completed')
confusion=confusion_matrix(y_test, predicted, labels=["steady", "righthand"])
print(confusion)

print(classification_report(y_test, predicted))

print('ok')