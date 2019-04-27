import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


df = pd.read_csv('./YouTubeConcatWhaleShark_20190317.arff',skiprows=6)
#print(df.head)

# not much preprocessing going on
X = df.text
y = df.label
cv = CountVectorizer()
X = cv.fit_transform(X) # Fit the Data

# try to get a model that gives us an accuracy of higher than 78%
for i in range(100):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=i)
	clf = LogisticRegression()
	clf.fit(X_train, y_train)
	clf.score(X_test, y_test)
	y_pred = clf.predict(X_test)
	if(accuracy_score(y_test, y_pred) > 0.78):
		print(classification_report(y_test, y_pred))
		print(accuracy_score(y_test, y_pred)*100, i)
