from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier

import torch
import warnings
import pandas as pd
import lightgbm as lgb

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    train = pd.read_csv('../Database/train_modified.csv', index_col='ID')
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
            'class_weight': {0: 8, 1: 5, 2: 2, 3: 1, 4: 1 / 3, 5: 1 / 3, 6: 1 / 3},
        }
        # {0: 8, 1: 5, 2: 2, 3: 1, 4: 1/3, 5: 1/3, 6: 1/3}
        rf_model = RandomForestClassifier(**params)
        rf_model.fit(X_train, y_train)
        print("Random Forest Accuracy:", accuracy_score(y_test, rf_model.predict(X_test)))

    Ada = False
    if Ada:
        params = {
            'base_estimator': DecisionTreeClassifier(max_depth=10),
            'n_estimators': 100,
            'learning_rate': 1,
            'algorithm': 'SAMME.R',
            'random_state': 42
        }

        ada_model = AdaBoostClassifier(**params)
        ada_model.fit(X_train, y_train)
        print("AdaBoost Accuracy:", accuracy_score(y_test, ada_model.predict(X_test)))

    GBM = False
    if GBM:
        params = {
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'num_class': 7,
            'device': 'cpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0
        }

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        num_round = 100
        bst = lgb.train(params, train_data, num_round, valid_sets=[valid_data])

        y_pred = bst.predict(X_test, num_iteration=bst.best_iteration)
        y_pred_class = [int(pred.argmax()) for pred in y_pred]

        accuracy = accuracy_score(y_test, y_pred_class)
        classification_rep = classification_report(y_test, y_pred_class)

        print(f"Accuracy: {accuracy}")
        print("Classification Report:\n", classification_rep)

    XGB = False
    if XGB:
        params = {'tree_method': 'gpu_hist', 'gpu_id': 0}
        xgb_model = XGBClassifier(n_estimators=700, random_state=2024, learning_rate=0.01, max_depth=80, **params)

        xgb_model.fit(X_train, y_train)
        print("XGBoost Accuracy:", accuracy_score(y_test, xgb_model.predict(X_test)))

    Cat = True
    if Cat:
        cat_model = CatBoostClassifier(random_state=2024,
                                       n_estimators=1000,
                                       learning_rate=0.01,
                                       depth=15,
                                       l2_leaf_reg=3,
                                       metric_period=1000,
                                       task_type='GPU')

        cat_features=[i for i in range(9,27)]

        cat_model.fit(X_train, y_train, cat_features=cat_features)
        print("CatBoost Accuracy:", accuracy_score(y_test, cat_model.predict(X_test)))

    VOTE = False
    if VOTE:
        voting_clf = VotingClassifier(estimators=[
            ('random_forest', rf_model),
            ('XGB_Boosting', xgb_model),
            ('CatBoosting', cat_model)
        ], voting='soft')

        voting_clf.fit(X_train, y_train)
        print("Voting Classifier Accuracy:", accuracy_score(y_test, voting_clf.predict(X_test)))

    Tab = False
    if Tab:

        model = TabNetClassifier(
            n_d=8,  # Decision 단계의 특성 차원
            n_a=8,  # Attention 단계의 특성 차원
            n_steps=5,  # Attention 단계의 반복 횟수
            gamma=1.5,  # Regularization 강도
            cat_idxs=[range(10,28)],  # 범주형 특성의 인덱스
            cat_dims=[],  # 범주형 특성의 차원
            cat_emb_dim=1,
            optimizer_fn=torch.optim.Adam,
        )

        model.fit(
            X_train.values, y_train,
            eval_set=[(X_train.values, y_train)],
            max_epochs=100,
            patience=20,
            batch_size=64,
            virtual_batch_size=32,
            num_workers=0,
            weights=1,
            drop_last=False,
            loss_fn=torch.nn.functional.cross_entropy,
        )

        preds = model.predict(X_test.values)
        print("Voting Classifier Accuracy:", accuracy_score(y_test, preds))
