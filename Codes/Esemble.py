from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from pytorch_tabnet.multitask import TabNetMultiTaskClassifier

import torch
import joblib
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
import torch.nn.functional as F


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
        cat_features = [i for i in range(7, 11)]
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

    def TabNet(self, params):
        bst = TabNetMultiTaskClassifier(**params)
        bst.fit(
            X_train=self.X_train.values,
            y_train=self.y_train.values.reshape(-1, 1),
            eval_set=[(self.X_test.values, self.y_test.values.reshape(-1, 1))],
            max_epochs=self.num_rounds,
            patience=15,
            loss_fn=F.cross_entropy,
            batch_size=2048,
            virtual_batch_size=1024,
        )

        y_pred = bst.predict_proba(self.X_test.values)
        y_pred = np.array(y_pred, dtype=np.float64).squeeze()
        predictions = y_pred.argmax(axis=1)
        accuracy = accuracy_score(self.y_test, predictions)

        joblib.dump(bst, '../Files/tab_model.pkl')
        print("Tabnet Accuracy:", accuracy)

        return accuracy if self.Tuning else y_pred

    def objective(self, trial):

        if self.method == 0:
            params = {
                'criterion': 'entropy',
                'class_weight': 'balanced',

                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 80),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0.01, 0.5),
                'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 10, 100),
            }
            accuracy = self.RandomForest(params)

        if self.method == 1:
            params = {
                'device': 'cpu',
                'boosting_type': 'gbrt',
                'objective': 'multiclass',
                'metric': 'multi_logloss',
                'tree_learner': 'voting',
                'num_class': 7,

                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'num_leaves': trial.suggest_int('num_leaves', 50, 300),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'subsample': trial.suggest_float('subsample', 0.01, 1),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1),
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
                'task_type': 'GPU',
                'boosting_type': 'Plain',
                'loss_function': 'MultiClass',
                'eval_metric': 'MultiClass',
                'classes_count': 7,

                'grow_policy': 'Lossguide',
                'bootstrap_type': 'Bayesian',
                'od_pval': 0.01,

                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5),
                'depth': trial.suggest_int('depth', 5, 20),
                'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 3, 20),
                'num_leaves': trial.suggest_int('num_leaves', 16, 300),
                'border_count': trial.suggest_int('border_count', 1, 300),
            }
            accuracy = self.CatBoost(params)

        if self.method == 4:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(device)

            params = {
                'verbose': True,
                'device_name': device,
                'cat_dims': [2, 11, 4, 12],
                'cat_idxs': [i for i in range(6, 10)],

                'n_d': trial.suggest_int('n_d', 20, 64),  # Decision 단계의 특성 차원
                'n_a': trial.suggest_int('n_a', 20, 64),  # Attention 단계의 특성 차원
                'n_steps': trial.suggest_int('n_steps', 10, 30),  # Attention 단계의 반복 횟수
                'gamma': trial.suggest_float('gamma', 0.8, 2),
            }
            accuracy = self.TabNet(params)

        return accuracy
