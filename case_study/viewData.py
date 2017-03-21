#!/usr/bin/python
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn  import preprocessing
train=pd.read_csv('/home/sky/Documents/case_study/training.csv',index_col=23)
test=pd.read_csv('/home/sky/Documents/case_study/testingCandidate.csv',index_col=21)


response=train['responded']
profit=train['profit']