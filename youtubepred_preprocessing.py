# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 14:32:07 2019

@author: joann
"""

import pandas as pd
import re
def process():
    df = pd.read_csv('C:/Users/joann/Documents/CS491 Machine Learning/shark/YouTubePredictor_20190319.arff',skiprows=8,sep="]',",engine='python')
    for i in range(len(df['text'])):
        df.at[i,'text'] = df.at[i,'text'].replace('\\',"")
        df.at[i,'text'] = df.at[i,'text'].replace('\'',"")
        df.at[i,'text'] = df.at[i,'text'].replace('\"',"")
        df.at[i,'text'] = re.sub(r"([.!?,'/()])", r" \1 ", df.at[i,'text'])
    return df   
