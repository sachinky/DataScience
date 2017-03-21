#!/usr/bin/python
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn  import preprocessing
train=pd.read_csv('/home/sky/Documents/case_study/training.csv',index_col=23)
test=pd.read_csv('/home/sky/Documents/case_study/testingCandidate.csv',index_col=21)


response=train['responded']
profit=train['profit']

df=train.ix[:,0:21]




def object2cat(f):
    object_cols=f.select_dtypes(['object']).columns
    f[object_cols]=f[object_cols].apply(lambda x:x.astype('category'))
    return f


def cat2dummies(f):
    category_cols=f.select_dtypes(include=['category']).columns
    ex_category_cols=f.select_dtypes(exclude=['category']).columns
    temp_df= pd.get_dummies(f[category_cols],prefix='is')
    #temp_df.append(f[ex_category_cols],axis=1)
    #return temp_df
    all_df=pd.concat([temp_df,f[ex_category_cols]],axis=1)
    return all_df
    
    
def cat2code(f):
    category_cols=f.select_dtypes(['category']).columns
    f[category_cols]=f[category_cols].apply(lambda x:x.cat.codes)
    return f
 


def preprocess(f,test_f):
    #convert object to categorical
    f=object2cat(f)
    test_f=object2cat(test_f)
    
    category_cols=f.select_dtypes(['category']).columns
    ex_category_cols=f.select_dtypes(exclude=['category']).columns
    
    #convert to codes
    f[category_cols]=f[category_cols].apply(lambda x:x.cat.codes)
    test_f[category_cols]=test_f[category_cols].apply(lambda x:x.cat.codes)

    
    #impute missing values
    imp=Imputer(missing_values='NaN',strategy='mean',axis=1)
    arr=imp.fit_transform(f)
    test_arr=imp.transform(test_f)
   
    df=pd.DataFrame(data=arr,index=f.index.tolist(),columns=list(f))
    test_df=pd.DataFrame(data=test_arr,index=test_f.index.tolist(),columns=list(test_f))
    
    #convert to dummies
    temp_df=pd.get_dummies(df[category_cols],prefix='is')
    all_df=pd.concat([temp_df,df[ex_category_cols]],axis=1)
    test_temp_df=pd.get_dummies(test_df[category_cols],prefix='is')
    test_all_df=pd.concat([test_temp_df,test_df[ex_category_cols]],axis=1)
    
    
    std_scale=preprocessing.StandardScaler().fit(all_df)
    scaled_df=std_scale.transform(all_df)
    scaled_test_df=std_scale.transform(test_all_df)
    
    
    return scaled_df,scaled_test_df
 
 
    
X_train,X_test=preprocess(df,test)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

Y_train=le.fit_transform(response)

from sklearn.decomposition import PCA
#PCA
pca = PCA(n_components=15).fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score

#LogisticRegression
from sklearn.linear_model import LogisticRegression
log=LogisticRegression(penalty='l2',C=0.01)


from sklearn.naive_bayes import GaussianNB
#Naive Bayes
gnb = GaussianNB()


#Random forest
from sklearn.ensemble import (RandomForestClassifier,GradientBoostingClassifier)
RFclf = RandomForestClassifier(n_estimators=20)

#SVM
from sklearn.svm import SVC
SVMclf=SVC(kernel='rbf',C=0.1)

#SGB
grd = GradientBoostingClassifier(n_estimators=20)


LRscores=cross_val_score(log,X_train,Y_train,cv=5)
GNBscores=cross_val_score(gnb,X_train,Y_train,cv=5)
RFscores=cross_val_score(RFclf,X_train,Y_train,cv=5)
SVMscores=cross_val_score(SVMclf,X_train,Y_train,cv=5)
SGBscores=cross_val_score(grd,X_train,Y_train,cv=5)
print LRscores.mean(),GNBscores.mean(),RFscores.mean(),SVMscores.mean(),SGBscores.mean()



Y_res=Y_train.nonzero()
X_train_pro = X_train[Y_res]
Y_train_pro=np.nan_to_num(profit)
Y_train_pro=Y_train_pro[Y_res]



from sklearn.linear_model import LinearRegression
lr=LinearRegression()
LRlabels=cross_val_predict(lr,X_train_pro,Y_train_pro,cv=5)
LRscores=cross_val_score(lr,X_train_pro,Y_train_pro,cv=5)
print LRscores.mean()


