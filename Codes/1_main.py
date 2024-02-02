from Esemble import *
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

import optuna
import warnings
import pandas as pd

if __name__ == "__main__":
    abcd = False
    if abcd:
        warnings.filterwarnings("ignore")
        train = pd.read_csv('../Database/train_abcd.csv', index_col='ID')
        X = train.drop(columns=['대출등급'])
        y = train['대출등급']
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # smote = SMOTE(random_state=42)
        # X_res, y_res = smote.fit_resample(X_train, y_train)

    if not abcd:
        warnings.filterwarnings("ignore")
        train = pd.read_csv('../Database/train_defg.csv', index_col='ID')
        X = train.drop(columns=['대출등급'])
        y = train['대출등급']
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    Tuning = False
    method = 2  # {RF=0, lightGBM=1, XGBoost=2, CatBoost=3}
    E = Esemble(method, X_train, X_val, y_train, y_val, 1000, Tuning, abcd)

    if method == 0:
        params = {
            'n_estimators': 100,
            'criterion': 'entropy',
            'max_depth': 80,
            'min_samples_split': 2,
            'min_samples_leaf': 2,
            'class_weight': 'balanced',
            'random_state': 42,
        }

        proba0 = E.RandomForest(params)

    if method == 1:
        if not Tuning:
            if abcd:
                params = {
                    'device': 'cpu',
                    'boosting_type': 'gbrt',
                    'objective': 'multiclass',
                    'metric': 'multi_logloss',
                    'tree_learner': 'voting',
                    'num_class': 4,

                    'learning_rate': 0.08748361678163433,
                    'max_depth':70,
                    'num_leaves': 261,
                    'min_data_in_leaf': 25,
                    'n_estimators': 266,
                    'subsample': 0.2615864200338772,
                    'colsample_bytree': 0.40973815068577557,
                }

            if not abcd:
                params = {
                    'device': 'cpu',
                    'boosting_type': 'gbrt',
                    'objective': 'multiclass',
                    'metric': 'multi_logloss',
                    'tree_learner': 'voting',
                    'num_class': 4,

                    'learning_rate': 0.23430996028026227,
                    'max_depth': 43,
                    'num_leaves': 151,
                    'min_data_in_leaf': 4,
                    'n_estimators': 392,
                    'subsample': 0.15935732666416363,
                    'colsample_bytree': 0.39844124393457364,
                }

            proba1 = E.lightGBM(params)

        if Tuning:
            study = optuna.create_study(direction='maximize')
            study.optimize(E.objective, n_trials=100)

            print('Number of finished trials:', len(study.trials))
            print('Best trial:', study.best_trial.params)

    if method == 2:
        if not Tuning:
            if abcd:
                params = {
                    'device': 'cuda',
                    'booster': 'gbtree',
                    'objective': 'multi:softprob',
                    'eval_metric': 'mlogloss',
                    'num_class': 4,

                    'eta': 0.19070557785560746,
                    'max_depth': 9,
                    'min_child_weight': 1,
                    'gamma': 0.10267706585587905,
                    'subsample': 0.943989141923496,
                    'colsample_bytree': 0.9719611765611176,
                    'colsample_bylevel': 0.7786561610050658,
                    'colsample_bynode': 0.9986410828042671,
                }

            if not abcd:
                params = {
                'device': 'cuda',
                'booster': 'gbtree',
                'objective': 'multi:softprob',
                'eval_metric': 'mlogloss',
                'num_class': 4,

                'eta': 0.19070557785560746,
                'max_depth': 9,
                'min_child_weight': 1,
                'gamma': 0.10267706585587905,
                'subsample': 0.943989141923496,
                'colsample_bytree': 0.9719611765611176,
                'colsample_bylevel': 0.7786561610050658,
                'colsample_bynode': 0.9986410828042671,
            }

            proba2 = E.XGBoost(params)

        if Tuning:
            study = optuna.create_study(direction='maximize')
            study.optimize(E.objective, n_trials=100)

            print('Number of finished trials:', len(study.trials))
            print('Best trial:', study.best_trial.params)

    if method == 3:
        if not Tuning:
            if abcd:
                params = {
                'task_type': 'GPU',
                'boosting_type': 'Plain',
                'loss_function': 'MultiClass',
                'eval_metric': 'MultiClass',
                'classes_count': 4,

                'grow_policy': 'Lossguide',
                'od_pval': 0.01,

                'learning_rate': 0.315293658365394,
                'depth': 21,
                'l2_leaf_reg': 17,
                'num_leaves': 245,
                'border_count': 278,
            }

            if not abcd:
                params = {
                    'task_type': 'GPU',
                    'boosting_type': 'Plain',
                    'loss_function': 'MultiClass',
                    'eval_metric': 'MultiClass',
                    'classes_count': 4,

                    'grow_policy': 'Lossguide',
                    'od_pval': 0.01,

                    'learning_rate': 0.3,
                    'depth': 21,
                    'l2_leaf_reg': 17,
                    'num_leaves': 245,
                    'border_count': 278,
                }

            proba3 = E.CatBoost(params)

        if Tuning:
            study = optuna.create_study(direction='maximize')
            study.optimize(E.objective, n_trials=100)

            print('Number of finished trials:', len(study.trials))
            print('Best trial:', study.best_trial.params)
