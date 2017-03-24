#!/usr/bin/python
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn  import preprocessing


%matplotlib inline

import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)


train=pd.read_csv('/home/sky/Documents/DataScience/case_study/training.csv',index_col=23)
test=pd.read_csv('/home/sky/Documents/DataScience/case_study/testingCandidate.csv',index_col=21)


response=train['responded']
profit=np.nan_to_num(train['profit'])

train.columns

#sns.swarmplot(x="custAge", y="profit", data=train, hue="responded");
#sns.boxplot(x="custAge", y="profit", data=train, hue="responded");

sns.countplot(x="euribor3m" ,data=train,hue="responded");
sns.swarmplot(x="responded", y="euribor3m", data=train);

sns.countplot(x=u'cons.price.idx' ,data=train,hue="responded");
sns.countplot(x=u'cons.conf.idx' ,data=train,hue="responded");
