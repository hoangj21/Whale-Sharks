# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 14:32:07 2019

@author: joann
"""

import pandas as pd
import re
def process():
    df = pd.read_csv('C:/Users/joann/Documents/CS491 Machine Learning/shark/YouTubePredictor_20190319.arff',skiprows=8,sep="]',",engine='python')
    counter = 0
    for sentence in df['text']:
        df.at[counter,'text'] = df.at[counter,'text'].replace('\\',"")
        df.at[counter,'text'] = df.at[counter,'text'].replace('\'',"")
        df.at[counter,'text'] = df.at[counter,'text'].replace('\"',"")
        df.at[counter,'text'] = re.sub(r"([.!?,'/()])", r" \1 ", df.at[counter,'text'])
        counter+=1
    return df   