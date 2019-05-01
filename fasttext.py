# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 21:40:51 2019

@author: joanna and josh
"""
import numpy as np
import ftfy
import codecs
import random
import fastText
from youtubepred_preprocessing import process

#for reporting accuracy data 
def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))
    return N,p,r

#for if we want to use the smaller concat dataset    
#df = pd.read_csv('./YouTubeConcatWhaleShark_20190317.arff',skiprows=6)

df = process() #preprocess data 

data = df.values #get array from dataframe 


#fix encoding of text
for each in data:
    each[0] = ftfy.fix_encoding(each[0])

train_output = codecs.open("fasttext_train.txt","w",encoding='utf-8') #opening file for training data
test_output = codecs.open("fasttext_test.txt","w",encoding='utf-8') #opening file for testing data

ft_line = "" #line to append fastText format onto later
percent_test_data =.2 #80/20 train test ratio used 

#reformatting data into fastText format 
for each in data:
    if type(each[1])==str: #ignore lines that don't have a label 
        fasttext_line = "__label__" + each[1] + " " +  str(each[0])
        if random.random() <= percent_test_data:
            test_output.write(fasttext_line + "\n")
        else:
            train_output.write(fasttext_line + "\n")
       
#closing files         
test_output.close()
train_output.close()  

classifier = fastText.train_supervised('fasttext_train.txt', epoch=25,wordNgrams=2, lr = .05) #train data
result = classifier.test('fasttext_test.txt') #test data

#getting accuracy measures 
N,p,r = print_results(*result)
fmeasure = 2*(p*r/(p+r))
print(fmeasure)

