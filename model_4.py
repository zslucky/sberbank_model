# coding=utf8
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

RS = 20170501
np.random.seed(RS)

ROUNDS = 500
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting': 'gbdt',
    'learning_rate': 0.04,
    'verbose': 0,
    'num_leaves': 2 ** 5,
    'bagging_fraction': 0.95,
    'bagging_freq': 0,
    'bagging_seed': RS,
    'feature_fraction': 0.7,
    'feature_fraction_seed': RS,
    'max_bin': 100,
    'max_depth': 5,
    'num_rounds': ROUNDS
}

print("Started")
input_folder = './'
train_df = pd.read_csv(input_folder + 'train.csv', parse_dates=['timestamp'])
test_df  = pd.read_csv(input_folder + 'test.csv' , parse_dates=['timestamp'])
macro_df = pd.read_csv(input_folder + 'macro.csv', parse_dates=['timestamp'])

bad_index = train_df[train_df.life_sq > train_df.full_sq].index
train_df.loc[bad_index, "life_sq"] = np.NaN
equal_index = [601,1896,2791]
test_df.loc[equal_index, "life_sq"] = test_df.loc[equal_index, "full_sq"]
bad_index = test_df[test_df.life_sq > test_df.full_sq].index
test_df.loc[bad_index, "life_sq"] = np.NaN
bad_index = train_df[train_df.life_sq < 5].index
train_df.loc[bad_index, "life_sq"] = np.NaN
bad_index = test_df[test_df.life_sq < 5].index
test_df.loc[bad_index, "life_sq"] = np.NaN
bad_index = train_df[train_df.full_sq < 5].index
train_df.loc[bad_index, "full_sq"] = np.NaN
bad_index = test_df[test_df.full_sq < 5].index
test_df.loc[bad_index, "full_sq"] = np.NaN
kitch_is_build_year = [13117]
train_df.loc[kitch_is_build_year, "build_year"] = train_df.loc[kitch_is_build_year, "kitch_sq"]
bad_index = train_df[train_df.kitch_sq >= train_df.life_sq].index
train_df.loc[bad_index, "kitch_sq"] = np.NaN
bad_index = test_df[test_df.kitch_sq >= test_df.life_sq].index
test_df.loc[bad_index, "kitch_sq"] = np.NaN
bad_index = train_df[(train_df.kitch_sq == 0).values + (train_df.kitch_sq == 1).values].index
train_df.loc[bad_index, "kitch_sq"] = np.NaN
bad_index = test_df[(test_df.kitch_sq == 0).values + (test_df.kitch_sq == 1).values].index
test_df.loc[bad_index, "kitch_sq"] = np.NaN
bad_index = train_df[(train_df.full_sq > 210) & (train_df.life_sq / train_df.full_sq < 0.3)].index
train_df.loc[bad_index, "full_sq"] = np.NaN
bad_index = test_df[(test_df.full_sq > 150) & (test_df.life_sq / test_df.full_sq < 0.3)].index
test_df.loc[bad_index, "full_sq"] = np.NaN
bad_index = train_df[train_df.life_sq > 300].index
train_df.loc[bad_index, ["life_sq", "full_sq"]] = np.NaN
bad_index = test_df[test_df.life_sq > 200].index
test_df.loc[bad_index, ["life_sq", "full_sq"]] = np.NaN
train_df.product_type.value_counts(normalize= True)
test_df.product_type.value_counts(normalize= True)
bad_index = train_df[train_df.build_year < 1500].index
train_df.loc[bad_index, "build_year"] = np.NaN
bad_index = test_df[test_df.build_year < 1500].index
test_df.loc[bad_index, "build_year"] = np.NaN
bad_index = train_df[train_df.num_room == 0].index
train_df.loc[bad_index, "num_room"] = np.NaN
bad_index = test_df[test_df.num_room == 0].index
test_df.loc[bad_index, "num_room"] = np.NaN
bad_index = [10076, 11621, 17764, 19390, 24007, 26713, 29172]
train_df.loc[bad_index, "num_room"] = np.NaN
bad_index = [3174, 7313]
test_df.loc[bad_index, "num_room"] = np.NaN
bad_index = train_df[(train_df.floor == 0).values * (train_df.max_floor == 0).values].index
train_df.loc[bad_index, ["max_floor", "floor"]] = np.NaN
bad_index = train_df[train_df.floor == 0].index
train_df.loc[bad_index, "floor"] = np.NaN
bad_index = train_df[train_df.max_floor == 0].index
train_df.loc[bad_index, "max_floor"] = np.NaN
bad_index = test_df[test_df.max_floor == 0].index
test_df.loc[bad_index, "max_floor"] = np.NaN
bad_index = train_df[train_df.floor > train_df.max_floor].index
train_df.loc[bad_index, "max_floor"] = np.NaN
bad_index = test_df[test_df.floor > test_df.max_floor].index
test_df.loc[bad_index, "max_floor"] = np.NaN
train_df.floor.describe(percentiles= [0.9999])
bad_index = [23584]
train_df.loc[bad_index, "floor"] = np.NaN
bad_index = train_df[train_df.state == 33].index
train_df.loc[bad_index, "state"] = np.NaN

# brings error down a lot by removing extreme price per sqm
train_df.loc[train_df.full_sq == 0, 'full_sq'] = 50
train = train_df[train_df.price_doc/train_df.full_sq <= 600000]
train = train_df[train_df.price_doc/train_df.full_sq >= 10000]

# Add month-year
month_year = (train_df.timestamp.dt.month + train_df.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
train_df['month_year_cnt'] = month_year.map(month_year_cnt_map)

month_year = (test_df.timestamp.dt.month + test_df.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
test_df['month_year_cnt'] = month_year.map(month_year_cnt_map)

# Add week-year count
week_year = (train_df.timestamp.dt.weekofyear + train_df.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
train_df['week_year_cnt'] = week_year.map(week_year_cnt_map)

week_year = (test_df.timestamp.dt.weekofyear + test_df.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
test_df['week_year_cnt'] = week_year.map(week_year_cnt_map)

# Add month and day-of-week
train_df['month'] = train_df.timestamp.dt.month
train_df['dow'] = train_df.timestamp.dt.dayofweek

test_df['month'] = test_df.timestamp.dt.month
test_df['dow'] = test_df.timestamp.dt.dayofweek

# Other feature engineering
train_df['rel_floor'] = train_df['floor'] / train_df['max_floor'].astype(float)
train_df['rel_kitch_sq'] = train_df['kitch_sq'] / train_df['full_sq'].astype(float)

test_df['rel_floor'] = test_df['floor'] / test_df['max_floor'].astype(float)
test_df['rel_kitch_sq'] = test_df['kitch_sq'] / test_df['full_sq'].astype(float)

train_df.apartment_name=train_df.sub_area + train_df['metro_km_avto'].astype(str)
test_df.apartment_name=test_df.sub_area + train_df['metro_km_avto'].astype(str)

train_df['room_size'] = train_df['life_sq'] / train_df['num_room'].astype(float)
test_df['room_size'] = test_df['life_sq'] / test_df['num_room'].astype(float)

'''
  Normalize housing pricing
'''
rate_2015_q2 = 1
rate_2015_q1 = rate_2015_q2 / .9932
rate_2014_q4 = rate_2015_q1 / 1.0112
rate_2014_q3 = rate_2014_q4 / 1.0169
rate_2014_q2 = rate_2014_q3 / 1.0086
rate_2014_q1 = rate_2014_q2 / 1.0126
rate_2013_q4 = rate_2014_q1 / 0.9902
rate_2013_q3 = rate_2013_q4 / 1.0041
rate_2013_q2 = rate_2013_q3 / 1.0044
rate_2013_q1 = rate_2013_q2 / 1.0104  # This is 1.002 (relative to mult), close to 1:
rate_2012_q4 = rate_2013_q1 / 0.9832  #     maybe use 2013q1 as a base quarter and get rid of mult?
rate_2012_q3 = rate_2012_q4 / 1.0277
rate_2012_q2 = rate_2012_q3 / 1.0279
rate_2012_q1 = rate_2012_q2 / 1.0279
rate_2011_q4 = rate_2012_q1 / 1.076
rate_2011_q3 = rate_2011_q4 / 1.0236
rate_2011_q2 = rate_2011_q3 / 1
rate_2011_q1 = rate_2011_q2 / 1.011


# train 2015
train_df['average_q_price'] = 1

train_2015_q2_index = train_df.loc[train_df['timestamp'].dt.year == 2015].loc[train_df['timestamp'].dt.month >= 4].loc[train_df['timestamp'].dt.month < 7].index
train_df.loc[train_2015_q2_index, 'average_q_price'] = rate_2015_q2

train_2015_q1_index = train_df.loc[train_df['timestamp'].dt.year == 2015].loc[train_df['timestamp'].dt.month >= 1].loc[train_df['timestamp'].dt.month < 4].index
train_df.loc[train_2015_q1_index, 'average_q_price'] = rate_2015_q1


# train 2014
train_2014_q4_index = train_df.loc[train_df['timestamp'].dt.year == 2014].loc[train_df['timestamp'].dt.month >= 10].loc[train_df['timestamp'].dt.month <= 12].index
train_df.loc[train_2014_q4_index, 'average_q_price'] = rate_2014_q4

train_2014_q3_index = train_df.loc[train_df['timestamp'].dt.year == 2014].loc[train_df['timestamp'].dt.month >= 7].loc[train_df['timestamp'].dt.month < 10].index
train_df.loc[train_2014_q3_index, 'average_q_price'] = rate_2014_q3

train_2014_q2_index = train_df.loc[train_df['timestamp'].dt.year == 2014].loc[train_df['timestamp'].dt.month >= 4].loc[train_df['timestamp'].dt.month < 7].index
train_df.loc[train_2014_q2_index, 'average_q_price'] = rate_2014_q2

train_2014_q1_index = train_df.loc[train_df['timestamp'].dt.year == 2014].loc[train_df['timestamp'].dt.month >= 1].loc[train_df['timestamp'].dt.month < 4].index
train_df.loc[train_2014_q1_index, 'average_q_price'] = rate_2014_q1


# train 2013
train_2013_q4_index = train_df.loc[train_df['timestamp'].dt.year == 2013].loc[train_df['timestamp'].dt.month >= 10].loc[train_df['timestamp'].dt.month <= 12].index
train_df.loc[train_2013_q4_index, 'average_q_price'] = rate_2013_q4

train_2013_q3_index = train_df.loc[train_df['timestamp'].dt.year == 2013].loc[train_df['timestamp'].dt.month >= 7].loc[train_df['timestamp'].dt.month < 10].index
train_df.loc[train_2013_q3_index, 'average_q_price'] = rate_2013_q3

train_2013_q2_index = train_df.loc[train_df['timestamp'].dt.year == 2013].loc[train_df['timestamp'].dt.month >= 4].loc[train_df['timestamp'].dt.month < 7].index
train_df.loc[train_2013_q2_index, 'average_q_price'] = rate_2013_q2

train_2013_q1_index = train_df.loc[train_df['timestamp'].dt.year == 2013].loc[train_df['timestamp'].dt.month >= 1].loc[train_df['timestamp'].dt.month < 4].index
train_df.loc[train_2013_q1_index, 'average_q_price'] = rate_2013_q1


# train 2012
train_2012_q4_index = train_df.loc[train_df['timestamp'].dt.year == 2012].loc[train_df['timestamp'].dt.month >= 10].loc[train_df['timestamp'].dt.month <= 12].index
train_df.loc[train_2012_q4_index, 'average_q_price'] = rate_2012_q4

train_2012_q3_index = train_df.loc[train_df['timestamp'].dt.year == 2012].loc[train_df['timestamp'].dt.month >= 7].loc[train_df['timestamp'].dt.month < 10].index
train_df.loc[train_2012_q3_index, 'average_q_price'] = rate_2012_q3

train_2012_q2_index = train_df.loc[train_df['timestamp'].dt.year == 2012].loc[train_df['timestamp'].dt.month >= 4].loc[train_df['timestamp'].dt.month < 7].index
train_df.loc[train_2012_q2_index, 'average_q_price'] = rate_2012_q2

train_2012_q1_index = train_df.loc[train_df['timestamp'].dt.year == 2012].loc[train_df['timestamp'].dt.month >= 1].loc[train_df['timestamp'].dt.month < 4].index
train_df.loc[train_2012_q1_index, 'average_q_price'] = rate_2012_q1


# train 2011
train_2011_q4_index = train_df.loc[train_df['timestamp'].dt.year == 2011].loc[train_df['timestamp'].dt.month >= 10].loc[train_df['timestamp'].dt.month <= 12].index
train_df.loc[train_2011_q4_index, 'average_q_price'] = rate_2011_q4

train_2011_q3_index = train_df.loc[train_df['timestamp'].dt.year == 2011].loc[train_df['timestamp'].dt.month >= 7].loc[train_df['timestamp'].dt.month < 10].index
train_df.loc[train_2011_q3_index, 'average_q_price'] = rate_2011_q3

train_2011_q2_index = train_df.loc[train_df['timestamp'].dt.year == 2011].loc[train_df['timestamp'].dt.month >= 4].loc[train_df['timestamp'].dt.month < 7].index
train_df.loc[train_2011_q2_index, 'average_q_price'] = rate_2011_q2

train_2011_q1_index = train_df.loc[train_df['timestamp'].dt.year == 2011].loc[train_df['timestamp'].dt.month >= 1].loc[train_df['timestamp'].dt.month < 4].index
train_df.loc[train_2011_q1_index, 'average_q_price'] = rate_2011_q1

train_df['price_doc'] = train_df['price_doc'] * train_df['average_q_price']

#fix outlier
# train_df.drop(train_df[train_df["life_sq"] > 5000].index, inplace=True)
mult = 1.054880504

train_y  = np.log(train_df['price_doc'].values * mult)
test_ids = test_df['id']

train_df.drop(['id', 'price_doc'], axis=1, inplace=True)
test_df.drop(['id'], axis=1, inplace=True)
print("Data: X_train: {}, X_test: {}".format(train_df.shape, test_df.shape))

df = pd.concat([train_df, test_df])

#Lets try using only those from https://www.kaggle.com/robertoruiz/sberbank-russian-housing-market/dealing-with-multicollinearity
macro_cols = ["timestamp","balance_trade","balance_trade_growth","eurrub","average_provision_of_build_contract","micex_rgbi_tr","micex_cbi_tr","deposits_rate","mortgage_value","mortgage_rate","income_per_cap","museum_visitis_per_100_cap","apartment_build"]
df = df.merge(macro_df[macro_cols], on='timestamp', how='left')
print("Merged with macro: {}".format(df.shape))

#Dates...
df['year'] = df.timestamp.dt.year
df['month'] = df.timestamp.dt.month
df['dow'] = df.timestamp.dt.dayofweek
df.drop(['timestamp'], axis=1, inplace=True)

#More featuers needed...

df_num = df.select_dtypes(exclude=['object'])
df_obj = df.select_dtypes(include=['object']).copy()
for c in df_obj:
    df_obj[c] = pd.factorize(df_obj[c])[0]

df_values = pd.concat([df_num, df_obj], axis=1)

pos = train_df.shape[0]
train_df = df_values[:pos]
test_df  = df_values[pos:]
del df, df_num, df_obj, df_values

print("Training on: {}".format(train_df.shape, train_y.shape))

train_lgb = lgb.Dataset(train_df, train_y)

cv_output = lgb.cv(params, train_lgb, num_boost_round=ROUNDS)
print(len(cv_output['rmse-mean']))

model = lgb.train(params, train_lgb, num_boost_round=ROUNDS)
preds = model.predict(test_df)

print("Writing output...")
out_df = pd.DataFrame({"id":test_ids, "price_doc":np.exp(preds)})
out_df.to_csv("lgb_{}_{}.csv".format(ROUNDS, RS), index=False)
# print(out_df.head(3))

# print("Features importance...")
# gain = model.feature_importance('gain')
# ft = pd.DataFrame({'feature':model.feature_name(), 'split':model.feature_importance('split'), 'gain':100 * gain / gain.sum()}).sort_values('gain', ascending=False)
# print(ft.head(25))

# plt.figure()
# ft[['feature','gain']].head(25).plot(kind='barh', x='feature', y='gain', legend=False, figsize=(10, 20))
# plt.gcf().savefig('features_importance.png')

# print("Done.")

train_pred = model.predict(train_df)
mse = mean_squared_error(train_y, train_pred)
print('mse = ', mse)
