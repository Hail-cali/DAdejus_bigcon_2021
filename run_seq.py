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
from utils.data_loader import *

def main():
    print(f"{'=' * 10} start grid search {'=' * 10}")
    start_time = time.time()
    opt = parse_opts()


    sample_models = ['rf', 'ada', 'xgb','sgdr','lgbm']

    print(f'- use model list {sample_models} -')

    ts = 10
    WINDOW_SIZE = 9

    opt = parse_opts()

    result = pd.read_csv(f'./dataset/seq_ts{ts}_ws{WINDOW_SIZE}.csv')

    X = result[['t' + str(i) for i in range(1, ts)]]
    y = result['t0']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

    mapped_model = {'xgb': ('xgboost', xgb.XGBRegressor()),
                    'lr': ('lr', lm.LinearRegression(n_jobs=-1)),
                    'sgdr': ('SGDRegressor', lm.SGDRegressor()),
                    'ada': ('AdaBoostRegressor', AdaBoostRegressor()),
                    'ridge': ('ridge', lm.Ridge()),
                    'lasso': ('lasso', lm.Lasso()),
                    'elastic': ('elastic', lm.ElasticNet()),
                    'LassoLars': ('LassoLars', lm.LassoLars()),
                    'logi': ('LogisticRegression', lm.LogisticRegression()),
                    'lgbm': ('LGBM', LGBMRegressor()),
                    'rf': ('RandomForest', RandomForestRegressor()),
                    }

    n = 3
    params = {
        'lr': {
            'fit_intercept': [True, False],
            'normalize': [True, False],
        },
        'ridge': {
            'alpha': [0.01, 0.1, 1.0, 10, 100],
            'fit_intercept': [True, False],
            'normalize': [True, False],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
        },
        'lasso': {
            'alpha': [0.1, 1.0, 10],
            'fit_intercept': [True, False],
            'normalize': [True, False],
            'selection': ['cyclic', 'random']
        },
        'elastic': {
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],  # default = 1.0
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],  # default = 0.5
            'fit_intercept': [True, False],  # default = True
            'normalize': [True, False],  # default = False
            'selection': ['cyclic', 'random']  # default = 'cyclic'
            # 'precompute': [True, False, 'auto'],  # default = False
            # 'copy_X': [True, False], #default = True
            # 'warm_start': [True, False], #default = False
            # 'positive': [True, False], #default = False
            # 'random_state': [None, 0, 42], #default = None
            # 'max_iter': [100, 500, 1000, 5000, 10000],  # default = 1000
            # 'tol': [0.0001, 0.001, 0.01, 0.1, 1],  # default = 0.0001
        },

        'LassoLars': {
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],  # default = 1.0
            'fit_intercept': [True, False],  # default = True
            'normalize': [True, False],  # default = True
            # 'verbose': [True, False],  # default = False
            # 'precompute': [True, False, 'auto'],  # default = 'auto'
            # 'eps': [2.220446049250313e-16,], #default = np.finfo(float).eps
            # 'copy_X': [True, False], #default = True
            # 'fit_path': [True, False], #default = True
            # 'positive': [True, False], #default = False
            # 'jitter': [None], #default = None
            # 'random_state': [None, 0, 42] #default = None
            # 'max_iter': [100, 500, 1000, 5000, 10000],  # default = 500
        },
        'LogisticRegression': {
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],  # default = 'l2'
            'C': [0.01, 0.1, 1.0, 10, 100],  # default = 1.0
            'fit_intercept': [True, False],  # default = True
            'class_weight': ['balanced', None],  # default = None
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],  # default = 'lbfgs'
            'max_iter': [10, 50, 100, 500, 1000, 5000],  # default = 100
            # 'random_state': [None, 0, 42], #default = None
            # 'multi_class': ['auto', 'ovr','multinomial'], #default = 'auto'
            # 'verbose': [0, 1, 2, 3, 4 ,5, 6, 7, 8, 9, 10], #default = 0
            # 'warm_start': [True, False], #default = False
            # 'n_jobs': [None, -1, 1], #default = None
            # 'l1_ratio': [None, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] #default = None
            # 'dual': [True, False], #default = False
            # 'tol': [0.0001, 0.001, 0.01, 0.1,1], #default = 0.0001
            # 'intercept_scaling': [0.1, 0.5, 1], #default = 1
        },

        'SGDRegressor': {
            'loss': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
            'epsilon': [0.1, 0.15, 0.2],
            # applies only when 'loss' parameter is 'huber', 'epsilon_insensitive', or 'squared_epsilon_insensitive'
            'penalty': ['l1', 'l2', 'elasticnet'],
            # 'l1_ratio':[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9], # applies only when 'penalty' parameter is set to 'elasticnet'
            # 'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100],
            'fit_intercept': [True, False],
            'learning_rate': ['optimal', 'constant', 'invscaling', 'adaptive'],
            'eta0': [0.0001, 0.001, 0.01],
            # applies only when 'learning rate' parameter is set to 'constant', 'invscaling', or 'adaptive'
            'power_t': [0.15, 0.25],
        },

        'xgboost': {
            # 'eta': [0.05, 0.1, 0.15, 0.2, 0.3], #default = 0.3, learning_rate, Typical values 0.01~0.2
            # "n_estimators": [70,100, 120],  # default = 100
            # 'max_depth': [3,5,6,7],  # default = 6, Typical values 3~10
            # 'min_child_weight': [1,2],  # default = 1
            # 'gamma': list(uniform(0, 0.5).rvs(n)), #default = 0
            # 'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], #default = 1, Typical values 0.5~1
            # 'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1], #default = 1, Typical values 0.5~1
            # 'scale_pos_weight': [1], #default = 1
            # 'objective': ['re'],
            # 'booster': ['gbdt', 'dart'],
            # 'seed': [2021] #default = 0,
            # 'nthread': -1,
            # 'max_delta_step': [0], #default = 0, this parameter is generally not used
            # 'sampling_method': ['uniform', 'gradient_based'], #default = uniform
            # 'colsample_bylevel': [0.5, 0.6, 0.7, 0.8, 0.9, 1], #default = 1, not used often
            # 'lambda': [1,2,3], #default = 1, to reduce overfitting, not used often
            # 'alpha': [0], #default = 0
            # "enable_categorical": [True],
        },


        'AdaBoostRegressor': {
            # 'n_estimators': [50, 100, 120],
            # 'learning_rate': [0.01, 0.05, 0.1, 0.3, 1],
            # 'loss': ['linear', 'square', 'exponential']
        },

        'LGBM': {
            # "learning_rate": [0.05, 0.04],
            # "max_bin": [512, 1000],
            # "num_leaves": [60, 80, 100, 110],
            # "min_data_in_leaf": [15, 18, 20],
            # 'min_data_in_leaf': [20],  # default = 100
            # 'boosting_type': ['gbdt', 'dart'],  # default = 'gbdt'
            # 'n_estimators': [100, 120],  # default = 100
            # 'objective': ['regression'],  # default = 'regression'
            # 'early_stopping_round': [50],  # default = 0
            # 'lambda_l1': #default = 0
            # 'lambda_l2': #default = 0
            # 'min_gain_to_split' #default = 0
            # 'num_iterations': [100], #default = 100
            # 'device': 'cpu', #default = 'cpu'인데 gpu이용 시 gpu로 지정
            # 'lambda'
            # 'feature_fraction'
            # 'bagging_fraction'
        },

        'RandomForest': {
            'n_estimators': [50, 100]
        },

    }

    models = [mapped_model[model] for model in sample_models]

    # print(models)

    best_model, best_mae, best_rmse = None, float('inf'), float('inf')

    level0 = list()
    level1 = lm.LinearRegression()

    for model_name, model in tqdm(models):
        param_grid = params[model_name]
        grid = GridSearchCV(model, cv=5, n_jobs=-1, param_grid=param_grid)

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

    # x_test = pd.read_csv(os.path.join(opt.data_path, 'baseball_test_final.csv'))
    # yhat = model.predict(x_test)
    #
    #
    # print(f"final predict value {[ name for name in  zip(x_test.NAME, yhat) ]}")


    end_time = time.time()
    print(f'from {opt.file}  take {end_time - start_time:0.3f} s')
    print(f"{'=' * 10} end gird search {'=' * 10}")
    return


if __name__ == '__main__':
    main()



