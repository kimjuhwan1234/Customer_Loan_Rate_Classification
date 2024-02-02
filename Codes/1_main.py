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

        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)

    if not abcd:
        warnings.filterwarnings("ignore")
        train = pd.read_csv('../Database/train_defg.csv', index_col='ID')
        X = train.drop(columns=['대출등급'])
        y = train['대출등급']
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)

    Tuning = False
    method = 3  # {RF=0, lightGBM=1, XGBoost=2, CatBoost=3}
    E = Esemble(method, X_res, X_val, y_res, y_val, 1000, Tuning, abcd)

    if method == 0:
        if not Tuning:
            params = {
                'criterion': 'entropy',
                'class_weight': 'balanced',

                'n_estimators': 100,
                'max_depth': 80,
                'min_samples_split': 2,
                'min_samples_leaf': 2,
                'max_leaf_nodes': 10,

                'random_state': 42,
            }
            proba0 = E.RandomForest(params)

        if Tuning:
            study = optuna.create_study(direction='maximize')
            study.optimize(E.objective, n_trials=100)

            print('Number of finished trials:', len(study.trials))
            print('Best trial:', study.best_trial.params)

    if method == 1:
        if not Tuning:
            params = {
                'device': 'cpu',
                'boosting_type': 'gbrt',
                'objective': 'multiclass',
                'metric': 'multi_logloss',
                'tree_learner': 'voting',
                'num_class': 4,

                'learning_rate': 0.07262236691173093,
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
