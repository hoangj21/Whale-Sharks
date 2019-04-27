# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 14:32:07 2019

@author: joann
"""

import pandas as pd

df = pd.read_csv('C:/Users/joann/Documents/CS491 Machine Learning/shark/YouTubePredictor_20190319.arff',skiprows=8,sep="]',",engine='python')
file = open("youtubepred.txt","w") 

counter = 0
for sentence in df['text']:
    df.at[counter,'text'] = df.at[counter,'text'].replace('[',"")
    df.at[counter,'text'] = df.at[counter,'text'].replace(']',"")
    df.at[counter,'text'] = df.at[counter,'text'].replace('\\',"")
    df.at[counter,'text'] = df.at[counter,'text'].replace('\'',"")
    df.at[counter,'text'] = df.at[counter,'text'].replace('\"',"")
    counter+=1
    file.write("%s\n" % sentence) 
   
print(df.columns)
print(df)        
