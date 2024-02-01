from Esemble import *
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

import optuna
import warnings
import pandas as pd

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    train = pd.read_csv('../Database/train_preprocessed.csv', index_col='ID')
    X = train.drop(columns=['대출등급'])
    y = train['대출등급']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    Tuning = True
    method = 2  # {RF=0, lightGBM=1, XGBoost=2, CatBoost=3}
    E = Esemble(method, X_res, X_val, y_res, y_val, 1000, Tuning)

    if method == 0:
        if not Tuning:
            params = {
                'criterion': 'entropy',
                'class_weight': 'balanced',

                'n_estimators': 100,
                'max_depth': 80,
                'min_samples_split': 2,
                'min_samples_leaf': 2,
                'min_weight_fraction_leaf': 1,
                'max_leaf_nodes': 10,
            }
            proba0 = E.RandomForest(params)

        if Tuning:
            study = optuna.create_study(direction='maximize')
            study.optimize(E.objective, n_trials=30)

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
                'num_class': 7,

                'learning_rate':0.10672172277306086,
                'max_depth': 24,
                'num_leaves': 104,
                'min_data_in_leaf': 15,
                'n_estimators': 476,
                'subsample': 0.5912715342643231,
                'colsample_bytree': 0.37175740831979154,
            }
            proba1 = E.lightGBM(params)

        if Tuning:
            study = optuna.create_study(direction='maximize')
            study.optimize(E.objective, n_trials=50)

            print('Number of finished trials:', len(study.trials))
            print('Best trial:', study.best_trial.params)

    if method == 2:
        if not Tuning:
            params = {
                'device': 'cuda',
                'booster': 'gbtree',
                'objective': 'multi:softprob',
                'eval_metric': 'mlogloss',
                'num_class': 7,

                'eta': 0.17649041654669656,
                'max_depth': 17,
                'min_child_weight': 4,
                'gamma': 0.7140173599495627,
                'subsample': 0.9427295598784581,
                'colsample_bytree': 0.7170099311806741,
                'colsample_bylevel': 0.956430558338542,
                'colsample_bynode': 0.9788900022922193,
            }
            proba2 = E.XGBoost(params)

        if Tuning:
            study = optuna.create_study(direction='maximize')
            study.optimize(E.objective, n_trials=30)

            print('Number of finished trials:', len(study.trials))
            print('Best trial:', study.best_trial.params)

    if method == 3:
        if not Tuning:
            params = {
                'task_type': 'GPU',
                'boosting_type': 'Plain',
                'loss_function': 'MultiClass',
                'eval_metric': 'MultiClass',
                'classes_count': 7,

                'bootstrap_type': 'Bayesian',
                'grow_policy': 'Lossguide',
                'od_pval': 0.01,

                'learning_rate': 0.3667491900056955,
                'depth': 28,
                'l2_leaf_reg': 9,
                'num_leaves': 137,
                'border_count': 277,
            }
            proba3 = E.CatBoost(params)

        if Tuning:
            study = optuna.create_study(direction='maximize')
            study.optimize(E.objective, n_trials=30)

            print('Number of finished trials:', len(study.trials))
            print('Best trial:', study.best_trial.params)
