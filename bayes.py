# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 14:43:03 2019

@author: joanna and josh
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

#skipping rows that don't contain data
df = pd.read_csv('C:/Users/joann/Documents/CS491 Machine Learning/shark/YouTubeConcatWhaleShark_20190317.arff',skiprows=6)

X = df['text'] #fetching text column
y = df['label'] #fetching labels 
df['label'].map({'poor': 0, 'good': 1}) #map labels to values 
cv = CountVectorizer() #for extracting word frequency vectors from text 
X = cv.fit_transform(X) #fitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #80/20 train/test ratio

#Naive Bayes Classifier
model = MultinomialNB()
model.fit(X_train,y_train)
model.score(X_test,y_test)
y_pred = model.predict(X_test)

#report accuracy and test metrics 
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))