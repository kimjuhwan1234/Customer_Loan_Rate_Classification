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
    method = 0  # {RF=0, lightGBM=1, XGBoost=2, CatBoost=3}
    E = Esemble(method, X_train, X_val, y_train, y_val, 2000, Tuning, abcd)

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

                    'learning_rate': 0.05030824092242113,
                    'max_depth': 30,
                    'num_leaves': 269,
                    'min_data_in_leaf': 5,
                    'n_estimators': 485,
                    'subsample': 0.36304547336992216,
                    'colsample_bytree': 0.605449704120681,
                }

            if not abcd:
                params = {
                    'device': 'cpu',
                    'boosting_type': 'gbrt',
                    'objective': 'multiclass',
                    'metric': 'multi_logloss',
                    'tree_learner': 'voting',
                    'num_class': 4,

                    'learning_rate': 0.05262236691173093,
                    'max_depth': 29,
                    'num_leaves': 269,
                    'min_data_in_leaf': 5,
                    'n_estimators': 483,
                    'subsample': 0.22577115327782551,
                    'colsample_bytree': 0.7247910478396337,
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

                    'eta': 0.05570506282108968,
                    'max_depth': 20,
                    'min_child_weight': 4,
                    'gamma': 0.4673653907355101,
                    'subsample': 0.8414340212323217,
                    'colsample_bytree': 0.6137340144642844,
                    'colsample_bylevel': 0.9874667930510251,
                    'colsample_bynode': 0.7785575663490459,
                }

            if not abcd:
                params = {
                'device': 'cuda',
                'booster': 'gbtree',
                'objective': 'multi:softprob',
                'eval_metric': 'mlogloss',
                'num_class': 4,

                'eta': 0.11570506282108968,
                'max_depth': 17,
                'min_child_weight': 4,
                'gamma': 0.4673653907355101,
                'subsample': 0.8414340212323217,
                'colsample_bytree': 0.6137340144642844,
                'colsample_bylevel': 0.9874667930510251,
                'colsample_bynode': 0.7785575663490459,
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

                'learning_rate': 0.05,
                'depth': 19,
                'l2_leaf_reg': 4,
                'num_leaves': 150,
                'border_count': 298,
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
                    'depth': 19,
                    'l2_leaf_reg': 4,
                    'num_leaves': 104,
                    'border_count': 298,
                }

            proba3 = E.CatBoost(params)

        if Tuning:
            study = optuna.create_study(direction='maximize')
            study.optimize(E.objective, n_trials=100)

            print('Number of finished trials:', len(study.trials))
            print('Best trial:', study.best_trial.params)
