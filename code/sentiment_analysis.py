import pandas as pd
import numpy as np

import re 
import string

import pickle

import nltk
from nltk import word_tokenize
from nltk import sentiment
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.classify import NaiveBayesClassifier

from sklearn.metrics import confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

# from google.cloud import language
# from google.cloud.language import enums
# from google.cloud.language import types

from confusion import print_confusion_matrix
import pickle

# Load pickle file with all IMDb reviews
tlj=pickle.load(open("../data/IMDb_TLJ_Reviews_v1.pickle", "rb" ) )

print(tlj)