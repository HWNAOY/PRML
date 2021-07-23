#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train = pd.read_csv("/kaggle/input/dataset/train.csv")
songs = pd.read_csv("/kaggle/input/dataset/songs.csv")
song_labels = pd.read_csv("/kaggle/input/dataset/song_labels.csv")
save = pd.read_csv("/kaggle/input/dataset/save_for_later.csv")
test = pd.read_csv("/kaggle/input/dataset/test.csv")


# In[ ]:


data = train


# In[ ]:


arr = np.zeros((14053,10000))
d = pd.DataFrame(arr, index = data['customer_id'].unique(), columns = data['song_id'].unique())


# In[ ]:


for index in range(len(data)):
    d.loc[data['customer_id'][index], data['song_id'][index]] = data['score'][index]


# In[ ]:





# In[ ]:


import numpy
import numpy as np
import pandas as pd

def matrix_factorization(R, P, Q, K, steps=200, alpha=0.0015, beta=0.15):
    Q = Q.T
    x = np.nonzero(R)
    for step in range(steps):
        for p in range(len(x[0])):  
            i = x[0][p]
            j = x[1][p]
            eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
            for k in range(K):
                P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = numpy.dot(P,Q)
        e = 0
        for p in range(len(x[0])):  
            i = x[0][p]
            j = x[1][p]
            e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
            for k in range(K):
                e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
            if e < 0.001:
                break
    return P, Q.T


# In[ ]:


R = np.array(d)
N = len(R)
M = len(R[0])
K = 2

P = np.random.rand(N,K)
Q = np.random.rand(M,K)

nP, nQ = matrix_factorization(R, P, Q, K)
nR = numpy.dot(nP, nQ.T)


# In[ ]:


dummy = pd.read_csv("/kaggle/input/prml-data-contest-jan-2021/dummy_submission.csv")


# In[ ]:


scores = []
for index in range(len(test)):
    c_id = test.iloc[index][0]
    s_id = test.iloc[index][1]
    i = d.index.get_loc(c_id)
    j = d.columns.get_loc(s_id)
    scores.append(nR[i][j])


# In[ ]:


result = dummy.copy()
result.drop('score', inplace = True, axis = 1)
result['score'] = scores


# In[ ]:





# In[ ]:


song_labels = song_labels.groupby("platform_id").agg({'label_id': lambda x: pd.Series.mode(x)[0],'count': 'mean'})


# In[ ]:


X1 = pd.merge(train,songs,on="song_id",how="left")
X1 = pd.merge(X1,song_labels,on="platform_id",how="left")

save["save_later"]=1
save = save.groupby('song_id').agg({'save_later':'sum'})

X1 = pd.merge(X1,save,on=["song_id"],how="left")


# In[ ]:


ave_user = train.groupby('customer_id').agg({'score':'mean'})
ave_user = ave_user.rename(columns={'score': 'ave_user'})

ave_item = train.groupby('song_id').agg({'score':'mean'})
ave_item = ave_item.rename(columns={'score': 'ave_item'})


# In[ ]:


X1 = pd.merge(X1,ave_user,on='customer_id',how='left')

X1 = pd.merge(X1,ave_item,on='song_id',how='left')


# In[ ]:


from sklearn.preprocessing import StandardScaler
cols_to_norm = ['released_year', 'number_of_comments','count','save_later'] #'label_id'
X1[cols_to_norm] = StandardScaler().fit_transform(X1[cols_to_norm])


# In[ ]:


X1['label_id']=X1['label_id'].fillna(X1['label_id'].mode().iloc[0])
X1['count']=X1['count'].fillna(X1['count'].mode().iloc[0])
X1['language']=X1['language'].fillna(X1['language'].mode().iloc[0])
X1['number_of_comments']=X1['number_of_comments'].fillna(X1['number_of_comments'].mode().iloc[0])
X1['released_year']=X1['released_year'].fillna(X1['released_year'].mode().iloc[0])
X1['save_later']=X1['save_later'].fillna(X1['save_later'].mode().iloc[0])

X1 = X1.drop(columns=['platform_id'])


# In[ ]:


X1['label_id'] = X1['label_id'].astype(int)

X1["customer_id"] = X1["customer_id"].astype('category')
X1["song_id"] = X1["song_id"].astype('category')

X1["language"] = X1["language"].astype('category')
X1["label_id"] = X1["label_id"].astype('category')


# In[ ]:


from catboost import CatBoostRegressor

features=['customer_id','song_id','released_year','language','number_of_comments','label_id','count','save_later']#'customer_id','song_id','platform_id',
c_features = ['customer_id','song_id','language','label_id']#'label_1','label_2','label_3'
y = X1.score 
X = X1[['customer_id','song_id','released_year','language','number_of_comments','label_id','count','save_later','ave_user','ave_item']] #X1


# In[ ]:


X_test = pd.merge(test,songs,on="song_id",how="left")
X_test = pd.merge(X_test,song_labels,on="platform_id",how="left")

X_test = pd.merge(X_test,save,on=["song_id"],how="left")


# In[ ]:


X_test = pd.merge(X_test,ave_user,on='customer_id',how='left')
X_test = pd.merge(X_test,ave_item,on='song_id',how='left')
X_test[cols_to_norm] = StandardScaler().fit_transform(X_test[cols_to_norm])


# In[ ]:


X_test['label_id']=X_test['label_id'].fillna(X_test['label_id'].mode().iloc[0])
X_test['count']=X_test['count'].fillna(X_test['count'].mode().iloc[0])
X_test['ave_user']=X_test['ave_user'].fillna(0)
X_test['ave_item']=X_test['ave_item'].fillna(0)
X_test['language']=X_test['language'].fillna(X_test['language'].mode().iloc[0])
X_test['number_of_comments']=X_test['number_of_comments'].fillna(X_test['number_of_comments'].mode().iloc[0])
X_test['released_year']=X_test['released_year'].fillna(X_test['released_year'].mode().iloc[0])
X_test['save_later']=X_test['save_later'].fillna(X_test['save_later'].mode().iloc[0])
X_test = X_test.drop(columns=['platform_id'])


# In[ ]:


X_test['label_id'] = X_test['label_id'].astype(int)

X_test["customer_id"] = X_test["customer_id"].astype('category')
X_test["song_id"] = X_test["song_id"].astype('category')

X_test["language"] = X_test["language"].astype('category')
X_test["label_id"] = X_test["label_id"].astype('category')


# In[ ]:


from catboost import Pool
from sklearn.model_selection import StratifiedKFold
model2 = CatBoostRegressor(depth=8,bagging_temperature=7.919,random_state=123)
# Create StratifiedKFold object.
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
predictions2 = []
   
for train_index, test_index in skf.split(X, y):
    X_train_fold,X_val_fold = X.loc[train_index,:],X.loc[test_index,:]
    y_train_fold,y_val_fold = y.loc[train_index],y.loc[test_index]
    eval_dataset = Pool(X_val_fold,y_val_fold,cat_features=c_features)
    model2.fit(X_train_fold, y_train_fold,cat_features=c_features,eval_set=eval_dataset,verbose=False)
    prediction = model2.predict(X_test)
    predictions2.append(prediction)


# In[ ]:


predictions_2=np.mean(predictions2,axis=0)


# In[ ]:


result2=pd.DataFrame({'test_row_id': test.index, 'score': predictions_2})


# In[ ]:


result['score'] = 0.3*result['score']
result2['score'] = 0.7*result2['score']


# In[ ]:


X2 = pd.DataFrame({'pred1':result.score,'pred2':result2.score})
X2['score'] = X2.sum(axis=1) 
output = pd.DataFrame({'test_row_id':result.test_row_id,'score':X2.score})


# In[ ]:


output.to_csv('my_final_contest.csv',index=False)


# In[ ]:




