#!/usr/bin/python
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn  import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer

g=open('/home/sky/Documents/categorisation/data/product_title.txt')
f=open('/home/sky/Documents/categorisation/data/label.txt')
labels=f.readlines()
products=g.readlines()

tfidf=TfidfVectorizer()
X=tfidf.fit_transform(products)



from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
Y=le.fit_transform(labels)




# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=0)




#Random forest
from sklearn.ensemble import (RandomForestClassifier,GradientBoostingClassifier)

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score



#classifier={RandomForestClassifier(),GradientBoostingClassifier(),xgb.XGBClassifier()}
clf=xgb.XGBClassifier()
clf.fit(X_train,y_train)
y_true, y_pred = y_test, clf.predict(X_test.toarray())
print(classification_report(y_true, y_pred))


