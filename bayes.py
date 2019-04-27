# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 14:43:03 2019

@author: joanna and josh
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

df = pd.read_csv('C:/Users/joann/Documents/CS491 Machine Learning/shark/YouTubeConcatWhaleShark_20190317.arff',skiprows=6)
print(df.columns)

X = df['text']
y = df['label']
df['label'].map({'poor': 0, 'good': 1})
cv = CountVectorizer()
X = cv.fit_transform(X) # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#Naive Bayes Classifier
clf = MultinomialNB()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))