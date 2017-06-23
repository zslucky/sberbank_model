import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing
import xgboost as xgb

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
id_test = test.id

mult = .969

# train.loc[train['life_sq'] > 850, 'life_sq'] = 850
# train.loc[train['num_room'] > 12.5, 'num_room'] = 12.5
# train.loc[train['kitch_sq'] > 300, 'kitch_sq'] = 300

y_train = train["price_doc"] * mult + 10
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

xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20, verbose_eval=5, show_stdv=False)
print('best num_boost_rounds = ', len(cv_output))
num_boost_rounds = len(cv_output)

# num_boost_rounds = 384  # This was the CV output, as earlier version shows
model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round= num_boost_rounds)

y_predict = model.predict(dtest)
model_2_output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})
model_2_output.to_csv('model_2_result.csv', index=False)