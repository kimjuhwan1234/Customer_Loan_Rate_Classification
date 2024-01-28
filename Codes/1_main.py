from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import joblib
import optuna
import warnings
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as cat


class Esemble:
    def __init__(self, method, X_train, X_val, y_train, y_val, num_rounds, Tuning: bool):
        self.method = method
        self.X_train = X_train
        self.X_test = X_val
        self.y_train = y_train
        self.y_test = y_val
        self.num_rounds = num_rounds
        self.Tuning = Tuning

    def RandomForest(self, params):
        rf_model = RandomForestClassifier(**params)
        rf_model.fit(self.X_train, self.y_train)
        y_pred = rf_model.predict_proba(self.X_test)
        predictions = y_pred.argmax(axis=1)
        accuracy = accuracy_score(self.y_test, predictions)

        joblib.dump(rf_model, '../Files/rf_model.pkl')
        print("RandomForest Accuracy:", accuracy)

        return accuracy if self.Tuning else y_pred

    def lightGBM(self, params):
        train_data = lgb.Dataset(self.X_train, label=self.y_train)
        valid_data = lgb.Dataset(self.X_test, label=self.y_test, reference=train_data)

        bst = lgb.train(params, train_data, self.num_rounds, valid_sets=[valid_data])
        y_pred = bst.predict(self.X_test, num_iteration=bst.best_iteration)
        predictions = [int(pred.argmax()) for pred in y_pred]
        accuracy = accuracy_score(self.y_test, predictions)

        joblib.dump(bst, '../Files/lgb_model.pkl')
        print("lightGBM Accuracy:", accuracy)

        return accuracy if self.Tuning else y_pred

    def XGBoost(self, params):
        train_data = xgb.DMatrix(self.X_train, label=self.y_train)
        valid_data = xgb.DMatrix(self.X_test, label=self.y_test)

        bst = xgb.train(params, train_data, self.num_rounds, evals=[(valid_data, 'eval')])
        X_test = xgb.DMatrix(self.X_test)
        y_pred = bst.predict(X_test)
        predictions = y_pred.argmax(axis=1)
        accuracy = accuracy_score(self.y_test, predictions)

        joblib.dump(bst, '../Files/xgb_model.pkl')
        print("XGBoost Accuracy:", accuracy)

        return accuracy if self.Tuning else y_pred

    def CatBoost(self, params):
        cat_features = [i for i in range(6, 10)]
        train_pool = cat.Pool(data=self.X_train, label=self.y_train, cat_features=cat_features)
        val_pool = cat.Pool(data=self.X_test, label=self.y_test, cat_features=cat_features)

        bst = cat.CatBoostClassifier(**params, iterations=self.num_rounds)
        bst.fit(train_pool, eval_set=val_pool, verbose=5)
        y_pred = bst.predict_proba(self.X_test)
        predictions = y_pred.argmax(axis=1)
        accuracy = accuracy_score(self.y_test, predictions)

        joblib.dump(bst, '../Files/cat_model.pkl')
        print("CatBoost Accuracy:", accuracy)

        return accuracy if self.Tuning else y_pred

    def objective(self, trial):

        if self.method == 0:
            params = {
                'device': 'cuda',
                'booster': 'gbtree',
                'objective': 'multi:softprob',
                'eval_metric': 'merror',
                'num_class': 7,

                'eta': trial.suggest_float('eta', 0.01, 0.5),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0.1, 5),
                'subsample': trial.suggest_float('subsample', 0.5, 1),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.01, 1),
                'colsample_bynode': trial.suggest_float('colsample_bynode', 0.01, 1),
            }
            accuracy = self.RandomForest(params)

        if self.method == 1:
            params = {
                'device': 'cuda',
                'booster': 'gbtree',
                'objective': 'multi:softprob',
                'eval_metric': 'merror',
                'num_class': 7,

                'eta': trial.suggest_float('eta', 0.01, 0.5),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0.1, 5),
                'subsample': trial.suggest_float('subsample', 0.5, 1),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.01, 1),
                'colsample_bynode': trial.suggest_float('colsample_bynode', 0.01, 1),
            }
            accuracy = self.lightGBM(params)

        if self.method == 2:
            params = {
                'device': 'cuda',
                'booster': 'gbtree',
                'objective': 'multi:softprob',
                'eval_metric': 'merror',
                'num_class': 7,

                'eta': trial.suggest_float('eta', 0.01, 0.5),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0.1, 5),
                'subsample': trial.suggest_float('subsample', 0.5, 1),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.01, 1),
                'colsample_bynode': trial.suggest_float('colsample_bynode', 0.01, 1),
            }
            accuracy = self.XGBoost(params)

        if self.method == 3:
            params = {
                'device': 'cuda',
                'booster': 'gbtree',
                'objective': 'multi:softprob',
                'eval_metric': 'merror',
                'num_class': 7,

                'eta': trial.suggest_float('eta', 0.01, 0.5),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0.1, 5),
                'subsample': trial.suggest_float('subsample', 0.5, 1),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.01, 1),
                'colsample_bynode': trial.suggest_float('colsample_bynode', 0.01, 1),
            }
            accuracy = self.CatBoost(params)

        return accuracy


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    original = pd.read_csv('../Database/train.csv')
    train = pd.read_csv('../Database/train_preprocessed.csv', index_col='ID')
    X = train.drop(columns=['대출등급'])
    y = train['대출등급']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    Tuning = True
    method = 2  # {RF=0, lightGBM=1, XGBoost=2, CatBoost=3}
    E = Esemble(method, X_train, X_val, y_train, y_val, 200, Tuning)

    if method == 0:
        params = {
            'n_estimators': 100,
            'criterion': 'entropy',
            'max_depth': 80,
            'min_samples_split': 2,
            'min_samples_leaf': 2,
            'class_weight': 'balanced',
        }

        if Tuning:
            study = optuna.create_study(direction='maximize')
            study.optimize(E.objective, n_trials=100)

            print('Number of finished trials:', len(study.trials))
            print('Best trial:', study.best_trial.params)

        if not Tuning:
            proba0 = E.RandomForest(params)

    if method == 1:
        params = {
            'device': 'cpu',
            'boosting_type': 'gbrt',
            'objective': 'multiclass',
            'metric': 'multi_logloss',

            'num_class': 7,
            'learning_rate': 0.05,
            'max_depth': 15,

            'num_leaves': 100,
            'min_data_in_leaf': 2,
            'tree_learner': 'voting',
        }

        if Tuning:
            study = optuna.create_study(direction='maximize')
            study.optimize(E.objective, n_trials=100)

            print('Number of finished trials:', len(study.trials))
            print('Best trial:', study.best_trial.params)

        if not Tuning:
            proba1 = E.lightGBM(params)

    if method == 2:
        params = {
            'device': 'cuda',
            'booster': 'gbtree',
            'objective': 'multi:softprob',
            'eval_metric': 'merror',
            'num_class': 7,

            'eta': 0.3,
            'max_depth': 15,
            'min_child_weight': 1,
            'gamma': 0.9486363409750519,
            'subsample': 0.7606871822635872,
            'colsample_bytree': 0.6187211406333024,
            'colsample_bylevel': 0.9696576494941724,
            'colsample_bynode': 0.8369204966889496,
        }

        if Tuning:
            study = optuna.create_study(direction='maximize')
            study.optimize(E.objective, n_trials=100)

            print('Number of finished trials:', len(study.trials))
            print('Best trial:', study.best_trial.params)

        if not Tuning:
            proba2 = E.XGBoost(params)

    if method == 3:
        # l2_leaf_reg=3,
        # boosting_type = 'Ordered'
        params = {
            'task_type': 'CPU',
            'booster': 'gbtree',
            'loss_function': 'MultiClass',
            'eval_metric': 'mlogloss',

            'num_class': 7,
            'learning_rate': 0.1,
            'depth': 10,

            'gamma': 3,
            'subsample': 0.6,
        }

        if Tuning:
            study = optuna.create_study(direction='maximize')
            study.optimize(E.objective, n_trials=100)

            print('Number of finished trials:', len(study.trials))
            print('Best trial:', study.best_trial.params)

        if not Tuning:
            proba3 = E.CatBoost(params)
