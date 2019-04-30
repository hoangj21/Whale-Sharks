# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 21:40:51 2019

@author: joann
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import ftfy
import codecs
import random
import fastText
import os
from youtubepred_preprocessing import process


def print_results(N, p, r):
    
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))
    return N,p,r
    
#df = pd.read_csv('C:/Users/joann/Documents/CS491 Machine Learning/shark/YouTubeConcatWhaleShark_20190317.arff',skiprows=6)
df = process()
#print(df)
data = df.values
for each in data:
    each[0] = ftfy.fix_encoding(each[0])

#print(data)
train_output = codecs.open("fasttext_train.txt","w",encoding='utf-8')
test_output = codecs.open("fasttext_test.txt","w",encoding='utf-8')
ft_line = ""
percent_test_data =.2
for each in data:
    if type(each[1])==str:
        fasttext_line = "__label__" + each[1] + " " +  str(each[0])
        if random.random() <= percent_test_data:
            test_output.write(fasttext_line + "\n")
        else:
            train_output.write(fasttext_line + "\n")
       
test_output.close()
train_output.close()  

classifier = fastText.train_supervised('fasttext_train.txt', epoch=25,wordNgrams=2, lr = .05)
#predict = classifier.predict('fasttext_test.txt')
#print(*predict)
result = classifier.test('fasttext_test.txt')
print(*result)
N,p,r = print_results(*result)
fmeasure = 2*(p*r/(p+r))
print(fmeasure)

