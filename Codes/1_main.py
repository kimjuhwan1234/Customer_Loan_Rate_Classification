from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import joblib
import warnings
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as cat


def RandomForest(params, X_train, y_train, X_test, y_test):
    rf_model = RandomForestClassifier(**params)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict_proba(X_test)
    predictions = y_pred.argmax(axis=1)

    joblib.dump(rf_model, '../Files/rf_model.pkl')
    print("RandomForest Accuracy:", accuracy_score(y_test, predictions))

    return y_pred


def lightGBM(params, X_train, y_train, X_test, y_test, num_rounds):
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    bst = lgb.train(params, train_data, num_rounds, valid_sets=[valid_data])
    y_pred = bst.predict(X_test, num_iteration=bst.best_iteration)
    predictions = [int(pred.argmax()) for pred in y_pred]

    joblib.dump(bst, '../Files/gbm_model.pkl')
    print("lightGBM Accuracy:", accuracy_score(y_test, predictions))

    return y_pred


def XGBoost(params, X_train, y_train, X_test, y_test, num_rounds):
    train_data = xgb.DMatrix(X_train, label=y_train)
    valid_data = xgb.DMatrix(X_test, label=y_test)

    bst = xgb.train(params, train_data, num_rounds, evals=[(valid_data, 'eval')])
    X_test = xgb.DMatrix(X_test)
    y_pred = bst.predict(X_test)
    predictions = y_pred.argmax(axis=1)

    joblib.dump(bst, '../Files/xgb_model.pkl')
    print("XGBoost Accuracy:", accuracy_score(y_test, predictions))

    return y_pred


def CatBoost(params, X_train, y_train, X_test, y_test, num_rounds):
    cat_features = [i for i in range(6, 10)]
    train_pool = cat.Pool(data=X_train, label=y_train, cat_features=cat_features)
    val_pool = cat.Pool(data=X_test, label=y_test, cat_features=cat_features)

    bst = cat.CatBoostClassifier(**params, iterations=num_rounds)
    bst.fit(train_pool, eval_set=val_pool, verbose=5)
    y_pred = bst.predict_proba(X_test)
    predictions = y_pred.argmax(axis=1)

    joblib.dump(bst, '../Files/cat_model.pkl')
    print("CatBoost Accuracy:", accuracy_score(y_test, predictions))

    return y_pred


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    original = pd.read_csv('../Database/train.csv')
    train = pd.read_csv('../Database/train_preprocessed.csv', index_col='ID')
    X = train.drop(columns=['대출등급'])
    y = train['대출등급']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    RF = False
    if RF:
        params = {
            'n_estimators': 100,
            'criterion': 'entropy',
            'max_depth': 80,
            'min_samples_split': 2,
            'min_samples_leaf': 2,
            'class_weight': 'balanced',
        }

        proba0 = RandomForest(params, X_train, y_train, X_test, y_test)

    GBM = False
    if GBM:
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

        proba1 = lightGBM(params, X_train, y_train, X_test, y_test, 200)

    XGB = False
    if XGB:
        params = {
            'device': 'cuda',
            'booster': 'gbtree',
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',

            'num_class': 7,
            'eta': 0.05,
            'max_depth': 15,

            'gamma': 3,
            'subsample': 0.6,
        }

        proba2 = XGBoost(params, X_train, y_train, X_test, y_test, 200)

    CAT = False
    if CAT:
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

        proba3 = CatBoost(params, X_train, y_train, X_test, y_test, 200)