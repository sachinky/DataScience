#!/usr/bin/python
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn  import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb

g=open('/home/sky/Documents/categorisation/data/product_title.txt')
f=open('/home/sky/Documents/categorisation/data/label.txt')
labels=f.readlines()
products=g.readlines()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
Y=le.fit_transform(labels)


from sklearn.feature_extraction.text import TfidfVectorizer

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    products, Y, test_size=0.7, random_state=0)

#Random forest
from sklearn.ensemble import (RandomForestClassifier,GradientBoostingClassifier)

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

#classifier={RandomForestClassifier(),GradientBoostingClassifier(),xgb.XGBClassifier()}
classifier={xgb.XGBClassifier()}

for c in classifier:
    pipeline=Pipeline([('tfidf',TfidfVectorizer()),('clf',c)])

    parameters={
    'tfidf__use_idf':(True,False),
    'tfidf__sublinear_tf':(True,False),
    'tfidf__ngram_range':((1,1),(1,2)),
    'tfidf__analyzer':('word','char'),
    'clf__n_estimators':(10,15,20,25),
    #'clf__criterion':("gini","entropy")
    
}



    grid_search=GridSearchCV(pipeline,parameters,verbose=1,cv=5)
    grid_search.fit(X_train,y_train)
    clf=grid_search
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))

