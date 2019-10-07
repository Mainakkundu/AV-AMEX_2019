# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 18:04:34 2019

@author: mainak.kundu
"""

import pandas as pd 
import numpy  as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
#%matplotlib inline
import os 

os.chdir(r'C:\Users\mainak.kundu\Desktop\AMEXPERT_2019')
train = pd.read_csv('train.csv')
cam = pd.read_csv('campaign_data.csv')
copun = pd.read_csv('coupon_item_mapping.csv')
demo = pd.read_csv('customer_demographics.csv')
trans = pd.read_csv('customer_transaction_data.csv')
item = pd.read_csv('item_data.csv')
test = pd.read_csv('test_QyjYwdj.csv')
sub = pd.read_csv('sample_submission_Byiv0dS (1).csv')


print('==train and campiagn merge==')
train_cam = train.merge(cam,on='campaign_id',how='left')
test_cam = test.merge(cam,on='campaign_id',how='left')

train_cam['IS_TRAIN'] = 'Y'
test_cam['IS_TRAIN']= 'N'
test_cam['redemption_status']= 0
f = [train_cam,test_cam]
full = pd.concat(f)
full.shape

full['start_date'] = pd.to_datetime(full['start_date'],infer_datetime_format=True)
full['end_date'] = pd.to_datetime(full['end_date'],infer_datetime_format=True)
full['GAP'] = full['end_date'] - full['start_date']
full['GAP'] = full['GAP']/np.timedelta64(1,'D')
full['GAP'] = np.abs(full['GAP'])


print('== Copun data(do an aggregation) + Full ==')
cpn_agg = copun.groupby('coupon_id')['item_id'].count().reset_index()
full_cop = full.merge(cpn_agg,on='coupon_id',how='inner')
sns.boxplot(full_cop['redemption_status'],full_cop['item_id'])

print('== Item Data & Copoun Data ==')
item_d = pd.get_dummies(item['category'],drop_first=True)
item_k = pd.get_dummies(item['brand_type'],drop_first=True)
item_f = pd.concat([item,item_d,item_k], axis=1)
del item_f['brand']
del item_f['category']
del item_f['brand_type']
copun_item = copun.merge(item_f,on='item_id',how='left')

copun_item_gr = copun_item.groupby('coupon_id').sum().reset_index()
del copun_item_gr['item_id']


print('== Customer Data ==')
full_cop_cust = full_cop.merge(demo,on='customer_id',how='left')
full_cop_cust[full_cop_cust['IS_TRAIN']=='N'].head()
len(full_cop_cust['id'].unique())
print('==Transaction Data ==')
trans['date'] = pd.to_datetime(trans['date'],infer_datetime_format=True)
trans['Month'] = pd.DatetimeIndex(trans['date']).month
v = trans.groupby('customer_id')['date'].count().reset_index()
trns_grp = trans.groupby('customer_id').sum()[['quantity','selling_price','other_discount','coupon_discount']].reset_index()
trns_grp['TOTAL_DISC'] = trns_grp['other_discount'] + trns_grp['coupon_discount']
trns_grp['CUST_VALUE'] = trns_grp['selling_price'] - trns_grp['TOTAL_DISC']
trns_grp = trns_grp.merge(v,on='customer_id')
full_cop_cust_trns = full_cop_cust.merge(trns_grp,on='customer_id',how='left')
f = full_cop_cust_trns.merge(copun_item_gr,on='coupon_id',how='left') ## final data 

sns.boxplot(full_cop_cust_trns['redemption_status'],full_cop_cust_trns['CUST_VALUE'])
full_cop_cust_trns.isnull().sum() # no null_values 
print('---- Base Line Data prepration done ----')
print('=========================================')

print('==Missing value treatment==')
full_cop_cust_trns.fillna(0,inplace=True)
full_cop_cust_trns

def binary_encoding(train_df, test_df, feat):
    # calculate the highest numerical value used for numeric encoding
    train_feat_max = train_df[feat].max()
    test_feat_max = test_df[feat].max()
    if train_feat_max > test_feat_max:
        feat_max = train_feat_max
    else:
        feat_max = test_feat_max
        
    # use the value of feat_max+1 to represent missing value
    train_df.loc[train_df[feat] == -1, feat] = feat_max + 1
    test_df.loc[test_df[feat] == -1, feat] = feat_max + 1
    
    # create a union set of all possible values of the feature
    union_val = np.union1d(train_df[feat].unique(), test_df[feat].unique())

    # extract the highest value from from the feature in decimal format.
    max_dec = union_val.max()
    
    # work out how the ammount of digtis required to be represent max_dev in binary representation
    max_bin_len = len("{0:b}".format(max_dec))
    index = np.arange(len(union_val))
    columns = list([feat])
    
    # create a binary encoding feature dataframe to capture all the levels for the feature
    bin_df = pd.DataFrame(index=index, columns=columns)
    bin_df[feat] = union_val
    
    # capture the binary representation for each level of the feature 
    feat_bin = bin_df[feat].apply(lambda x: "{0:b}".format(x).zfill(max_bin_len))
    
    # split the binary representation into different bit of digits 
    splitted = feat_bin.apply(lambda x: pd.Series(list(x)).astype(np.uint8))
    splitted.columns = [feat + '_bin_' + str(x) for x in splitted.columns]
    bin_df = bin_df.join(splitted)
    
    # merge the binary feature encoding dataframe with the train and test dataset - Done! 
    train_df = pd.merge(train_df, bin_df, how='left', on=[feat])
    test_df = pd.merge(test_df, bin_df, how='left', on=[feat])
    return train_df, test_df




train = f[f['IS_TRAIN']=='Y']
test = f[f['IS_TRAIN']=='N']



print('=== Categorical Handling===')
age = {0:0,'18-25':1,'26-35':2,'36-45':3,'46-55':4,'56-70':5,'70+':6}
train['age_range'] = train['age_range'].map(age)  
test['age_range'] = test['age_range'].map(age)  
camp = {'X':0,'Y':1}   
mar = {'Married':2,'Single':1}
train['marital_status'].fillna(0,inplace=True) 
test['marital_status'].fillna(0,inplace=True)  
cat_cols = ['campaign_type', 'age_range', 'marital_status',
       'family_size', 'no_of_children']

train['campaign_type'] = train['campaign_type'].map(camp)
test['campaign_type'] = test['campaign_type'].map(camp) 
#train['marital_status'] = train['marital_status'].map(mar)
#test['marital_status'] = test['marital_status'].map(mar)
train, test=binary_encoding(train, test, 'campaign_type') 

del train['campaign_type']
del test['campaign_type']
del train['marital_status']
del test['marital_status']


FEATURES =['GAP','item_id','age_range','rented','income_bracket','quantity','selling_price',
           'other_discount','CUST_VALUE', 'campaign_type_bin_0','date','Bakery', 'Dairy, Juices & Snacks',
       'Flowers & Plants', 'Fuel', 'Garden', 'Grocery', 'Meat',
       'Miscellaneous', 'Natural Products', 'Packaged Meat', 'Pharmaceutical',
       'Prepared Food', 'Restauarant', 'Salads', 'Seafood', 'Skin & Hair Care',
       'Travel', 'Vegetables (cut)', 'Local']
DEPEN =['redemption_status']
X = train[FEATURES]
y = train[DEPEN]
X_test = test[FEATURES]

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import time

X_train, X_val,y_train,y_val = train_test_split(X,y,test_size=0.33, random_state=42)

rf=RandomForestClassifier(n_estimators=300, n_jobs=6, min_samples_split=5, max_depth=7,
                          criterion='gini', random_state=0)


rf.fit(X_train,y_train)
y_pred = rf.predict(X_val)
y_pred_prob = rf.predict_proba(X_val)[:,1]
print(roc_auc_score(y_val,y_pred_prob)) ## 0.87 

rf.fit(X,y)
y_test_pred = rf.predict_proba(X_test)[:,1]
test['redemption_status'] = y_test_pred

scaler = StandardScaler().fit(X_train.values)
X_train_scl = scaler.transform(X_train)
X_val_scl = scaler.transform(X_val)
X_test = scaler.transform(X_test)
logit=LogisticRegression()
logit.fit(X_train,y_train)
y_pred = logit.predict(X_val)
y_pred_prob = logit.predict_proba(X_val)[:,1]
print(roc_auc_score(y_val,y_pred_prob)) ##0.56

print('==Extra Tree==')
et=RandomForestClassifier(n_estimators=100, n_jobs=6, min_samples_split=5, max_depth=5,
                          criterion='gini', random_state=0)
et.fit(X_train,y_train)
y_pred = et.predict(X_val)
y_pred_prob = et.predict_proba(X_val)[:,1]
print(roc_auc_score(y_val,y_pred_prob)) ## 0.855

print('===Neural Nets ===')









