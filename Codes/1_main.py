from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
# from pytorch_tabnet.multitask import TabNetMultiTaskClassifier

# import torch
import joblib
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb


# import torch.nn.functional as F

def AdaBoost(params, X_train, y_train, X_test, y_test):
    ada_model = AdaBoostClassifier(**params)
    ada_model.fit(X_train, y_train)
    print("AdaBoost Accuracy:", accuracy_score(y_test, ada_model.predict(X_test)))


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


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    original = pd.read_csv('../Database/train.csv')
    train = pd.read_csv('../Database/train_preprocessed.csv', index_col='ID')
    X = train.drop(columns=['대출등급'])
    y = train['대출등급']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    Ada = True
    if Ada:
        params = {
            'base_estimator': DecisionTreeClassifier(max_depth=5),
            'n_estimators': 100,
            'learning_rate': 1,
            'algorithm': 'SAMME.R',
            'random_state': 42
        }

        AdaBoost(params, X_train, y_train, X_test, y_test)

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


    Tab = False
    if Tab:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        y_train = np.reshape(y_train, (77034, 1))
        y_test = np.reshape(y_test, (19259, 1))

        model = TabNetMultiTaskClassifier(
            n_d=8,  # Decision 단계의 특성 차원
            n_a=8,  # Attention 단계의 특성 차원
            n_steps=5,  # Attention 단계의 반복 횟수
            gamma=1.5,  # Regularization 강도
            cat_idxs=[i for i in range(9, 13)],  # 범주형 특성의 인덱스
            cat_dims=[2, 16, 4, 12],  # 범주형 특성의 차원
            device_name=device,
        )

        model.fit(
            X_train=X_train.values,
            y_train=y_train,
            eval_set=[(X_test.values, y_test)],
            max_epochs=2,
            patience=5,
            loss_fn=F.cross_entropy,
            batch_size=64,
            virtual_batch_size=32,
        )

        preds = model.predict(X_test.values)
        print("TabNet Classifier Accuracy:", accuracy_score(y_test, preds))
