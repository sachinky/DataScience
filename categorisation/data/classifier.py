#!/usr/bin/python
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn  import preprocessing
from sklearn.metrics import classification_report
import sklearn.metrics as metric

import matplotlib.pyplot as plt


import seaborn as sns
sns.set(style="whitegrid", color_codes=True)


g=open('product_title.txt')
f=open('label.txt')
labels=f.readlines()
products=g.readlines()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
Y_train=le.fit_transform(labels)

sns.countplot(x=Y_train)

from sklearn.feature_extraction.text import TfidfVectorizer
tf_vect = TfidfVectorizer(use_idf=False,sublinear_tf=True,ngram_range=(1,2), analyzer='word').fit(products)
X_train_tf = tf_vect.transform(products)
X_train_tf.shape

from sklearn.naive_bayes import MultinomialNB
#Random forest
from sklearn.ensemble import (RandomForestClassifier,GradientBoostingClassifier)
RFclf = RandomForestClassifier(n_estimators=50)
#SGB
grd = GradientBoostingClassifier(n_estimators=10)

from sklearn.svm import SVC
SVMclf=SVC(kernel='linear',C=10)

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

clf = MultinomialNB()
NB_scores=cross_val_score(clf,X_train_tf,Y_train,cv=5)
RF_scores=cross_val_score(RFclf,X_train_tf,Y_train,cv=5)
#SGD_scores=cross_val_score(grd,X_train_tf.toarray(),Y_train,cv=5)
SVM_scores=cross_val_score(SVMclf,X_train_tf,Y_train,cv=5)
print NB_scores.mean(),RF_scores.mean(),SVM_scores.mean()
Y_pred=cross_val_predict(SVMclf,X_train_tf,Y_train,cv=5)

print(classification_report(Y_train, Y_pred))

