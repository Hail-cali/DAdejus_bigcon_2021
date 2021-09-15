# 모듈 임포트

import os
import sys

WORKING_DIR_AND_PYTHON_PATHS = os.path.join('/', *os.getcwd().split("/"))

sys.path.append(WORKING_DIR_AND_PYTHON_PATHS)


import warnings
import pandas as pd

from random import uniform
from random import randint

from tqdm import tqdm

warnings.filterwarnings(action='ignore')

from opt import *
import time

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import GridSearchCV
from sklearn import linear_model as lm
import xgboost as xgb
from scipy.stats import uniform, randint
from sklearn.ensemble import AdaBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def main():
    print(f"{'=' * 10} start grid search {'=' * 10}")
    start_time = time.time()
    opt = parse_opts()

    if not opt.models:
        opt.models = ['rf', 'xgb','lgbm']

    print(f'- use model list {opt.models} -')

    try:

        dataset = pd.read_csv(os.path.join(opt.data_path, opt.file))
        print(f'dataset shape: {dataset.shape}')
        print(f'dataset columns {dataset.columns.to_list()}')
    except:
        print(f'<PathErr> check file path :{os.path.join(opt.data_path, opt.file)}')
        dataset = None

    if opt.file:

        X_feature = ['선발', '타수', '득점', '안타', '2타', '3타', '홈런', '루타',
                     '타점', '도루', '도실', '볼넷', '사구', '고4', '삼진', '병살',
                     '희타', '희비', '투구', 'barrel', '타율', 'LG', 'KIA', 'KT',
                     '키움', '두산', '한화', 'NC', '롯데', '삼성', 'SSG', '홈경기수',
                     '원정경기수']


    else:
        X_feature = []


    print(f'used x feature {X_feature}')
    y_feature = opt.y_feature

    X = dataset[X_feature]
    y = dataset.loc[:, y_feature]

    if 'PCODE' in X_feature:
        X['PCODE'] = X['PCODE'].astype('category')

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

    if opt.modeltype == 'ensemble':

        mapped_model = {'xgb': ('xgboost', xgb.XGBRegressor()),
                        'lgbm': ('LGBM', LGBMRegressor()),
                        'rf': ('RandomForest', RandomForestRegressor()),
                        }
        if opt.y_feature == '장타':
            params = {
            'xgboost': {'eta' : [0.15],
                            'gamma' : [0.5],
                            'lambda' :[1] ,
                            'max_depth' : [5],
                            'min_child_weight' : [1],
                            'n_estimators' :[100],
                            'subsample' : [0.9]
            },


            'LGBM': {'lambda_l1':[1],
                     'lambda_l2':[1.5],
                     'min_data_in_leaf':[70],
                     'num_leaves':[4]
            },

            'RandomForest': {
                'n_estimators':[100],
                'max_depth':[5],
                'min_samples_leaf':[22],
                'min_samples_split':[6]
            },

        }
        elif opt.y_feature == '출루':
            params = {
                'xgboost': {'eta' : [0.15],
                            'gamma' : [0.5],
                            'lambda' :[1] ,
                            'max_depth' : [5],
                            'min_child_weight' : [1],
                            'n_estimators' :[100],
                            'subsample' : [0.9]
                },

                'LGBM': {'lambda_l1':[0.5],
                         'lambda_l2':[2.5],
                         'min_data_in_leaf':[90],
                         'num_leaves':[3]

                },

                'RandomForest': {
                    'n_estimators':[100],
                    'max_depth':[4],
                    'min_samples_leaf':[9],
                    'min_samples_split':[4]
                },

            }

        models = [mapped_model[model] for model in opt.models]


        best_model, best_mae, best_rmse = None, float('inf'), float('inf')

        level0 = list()
        level1 = lm.LinearRegression()

        for model_name, model in tqdm(models):
            param_grid = params[model_name]
            grid = GridSearchCV(model, cv=2, n_jobs=-1, param_grid=param_grid)

            grid = grid.fit(X_train, y_train)

            model = grid.best_estimator_
            predictions = model.predict(X_val)
            mae = mean_absolute_error(y_val, predictions)
            rmse = mean_squared_error(y_val, predictions) ** 0.5

            level0.append((f'{model_name}', model))

            print(f"{'-'*5:<40}{'-'*5:>40}");print(f"{'|':<40}{'|':>40}")
            print(f" model name | {model_name} | MAE: {mae} RMSE: {rmse} \n\n best params {model}  ")
            print(f"{'|':<40}{'|':>40}");print(f"{'-' * 5:<40}{'-' * 5:>40}")

            if mae < best_mae:
                best_model = model
                best_mae = mae
                best_rmse = rmse

        print(f"{'=' * 25}'total best model {'=' * 25}\n")
        print(f'used model {opt.models}')
        print(f'total best model: {best_model} params: mae {best_mae} rmse {best_rmse}\n')
        print(f"{'=' * 50}\n")


    model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)
    model.fit(X, y)
    x_test = pd.read_csv(os.path.join(opt.data_path, 'baseball_test_final_ip.csv'))
    y_hat = model.predict(x_test[X_feature])

    best_model.fit(X,y)
    best_y=best_model.predict(x_test[X_feature])
    print(f"best model predict value {[name for name in zip(x_test.NAME, best_y)]}")
    y_true = []
    print(f"final predict value {[ name for name in  zip(x_test.NAME, y_hat) ]}")
    # test_mae = mean_absolute_error(y_hat, y_true)
    # print(f'test_mae')

    end_time = time.time()
    print(f'from {opt.file} y_feature {y_feature}  take {end_time - start_time:0.3f} s')
    print(f"{'=' * 10} end gird search {'=' * 10}")
    return


if __name__ == '__main__':
    main()



