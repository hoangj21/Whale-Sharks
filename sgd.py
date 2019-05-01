# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 14:43:03 2019
@author: joanna and josh
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv('./YouTubeConcatWhaleShark_20190317.arff',skiprows=6)
print(df.columns)

X = df.text
y = df.label
cv = CountVectorizer()
X = cv.fit_transform(X) # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)
sgd.fit(X_train, y_train)


y_pred = sgd.predict(X_test)
print(classification_report(y_test, y_pred))