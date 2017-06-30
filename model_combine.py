import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing
import xgboost as xgb

model_1_result = pd.read_csv('./model_1_result.csv')
model_2_result = pd.read_csv('./model_2_result.csv')
model_3_result = pd.read_csv('./model_3_result.csv')

model_4_result = pd.read_csv('./lgb_500_20170501.csv')

first_result = model_2_result.merge(model_3_result, on="id", suffixes=['_louis','_bruno'])
first_result["price_doc"] = np.exp( .714*np.log(first_result.price_doc_louis) +
                                    .286*np.log(first_result.price_doc_bruno) )

second_result = first_result.merge(model_4_result, on="id", suffixes=['_first','_second'])
second_result["price_doc"] = np.exp( 1*np.log(second_result.price_doc_first) +
                                     0*np.log(second_result.price_doc_second) )

result = second_result.merge(model_1_result, on="id", suffixes=['_follow','_gunja'])

result["price_doc"] = np.exp( .7*np.log(result.price_doc_follow) +
                              .3*np.log(result.price_doc_gunja) )

result["price_doc"] =result["price_doc"] # * 0.9915
result.drop(["price_doc_louis","price_doc_bruno","price_doc_follow","price_doc_gunja","price_doc_first","price_doc_second"],axis=1,inplace=True)
result.head()
result.to_csv('sub_combine_result.csv', index=False)