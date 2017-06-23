import numpy as np
import pandas as pd
from sklearn import preprocessing
import lightgbm as lgb

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

y_train = train["price_doc"]
x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)
x_test = test.drop(["id", "timestamp"], axis=1)

for c in x_train.columns:
    if x_train[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_train[c].values))
        x_train[c] = lbl.transform(list(x_train[c].values))

for c in x_test.columns:
    if x_test[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_test[c].values))
        x_test[c] = lbl.transform(list(x_test[c].values))


train_data = lgb.Dataset(x_train, label=y_train)


param = {'num_leaves': 3, 'num_trees': 500, 'objective': 'huber'}
param['metric'] = 'l2_root'

num_round = 1000
output_cv = lgb.cv(param, train_data, num_round, nfold=5)


print(output_cv)