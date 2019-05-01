# Is That a Whale Shark?
### CS 491 Intro to Machine Learning final project- classifying youtube videos by using natural language processing on youtube titles/tags/descriptions to determine whether or not a video is about whale sharks 
#### By Joanna Hoang and Joshua Azicate

Note: original data not included in repo for privacy of original data owners                                              
## For our code to run properly, the following modifications must be made in the original datasets:

In the YouTubeConcatWhaleShark file below the @data located on line 6, add the following line:  
 text,label
 
In the YoutubeWhaleShark YouTubePredictor file below the @data located on line 8, add the following line:
 text]',label
 
We also removed the last two lines in the YouTubePredictor file which reported the 'poor' and 'good counts of the file.  

Change path names as necessary at the pd.read_csv lines in the files youtubepred_preprocessing.py, bayes.py, linearSVC.py,logisticregression.py

## File compatibility: 
bayes.py, linearSVC.py, and logisticregression.py were written to work with the YoutubeConcatWhaleShark file
youtubepred_preprocesssing.py in combination with fasttext.py were written to work with the YouTubePredictor file
