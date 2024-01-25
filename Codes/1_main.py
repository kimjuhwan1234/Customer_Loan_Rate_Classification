from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
# from pytorch_tabnet.multitask import TabNetMultiTaskClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder

# import torch
import joblib
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
# import torch.nn.functional as F

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    original= pd.read_csv('../Database/train.csv')
    train = pd.read_csv('../Database/train_modified4.csv', index_col='ID')
    test = pd.read_csv('../Database/test_modified4.csv', index_col='ID')
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

    GBM = True
    if GBM:
        params = {
            'learning_rate': 0.05,
            'boosting_type': 'gbrt',
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'num_class': 7,
            'device': 'cpu',
            'num_leaves': 100,
            'max_depth': 20,
            'min_data_in_leaf': 2,
            'tree_learner': 'voting',
        }

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        num_round = 100
        bst = lgb.train(params, train_data, num_round, valid_sets=[valid_data])
        y_pred = bst.predict(X_test, num_iteration=bst.best_iteration)
        y_pred_class = [int(pred.argmax()) for pred in y_pred]

        joblib.dump(bst, '../Files/gbm_model.pkl')
        print("lightGBM Accuracy:", accuracy_score(y_test, y_pred_class))

    XGB = False
    if XGB:
        params = {'tree_method': 'gpu_hist', 'gpu_id': 0}
        xgb_model = XGBClassifier(n_estimators=1000, random_state=2024, learning_rate=0.01, max_depth=20, **params)

        xgb_model.fit(X_train, y_train)

        joblib.dump(xgb_model, '../Files/xgb_model.pkl')
        print("XGBoost Accuracy:", accuracy_score(y_test, xgb_model.predict(X_test)))

    VOTE = False
    if VOTE:
        xgb_model = joblib.load('../Files/xgb_model.pkl')
        cat_model = joblib.load('../Files/cat_model.pkl')
        # gbm_model = joblib.load('../Files/gbm_model.pkl')

        proba1 = xgb_model.predict_proba(X_test)
        proba2 = cat_model.predict_proba(X_test)
        # proba3 = gbm_model.predict(X_test, num_iteration=gbm_model.best_iteration)

        xgb_result = np.argmax(proba1, axis=1)
        cat_result = np.argmax(proba2, axis=1)
        # gbm_result = [int(pred.argmax()) for pred in proba3]

        print("XGBoost Accuracy:", accuracy_score(y_test, xgb_result))
        print("CatBoost Accuracy:", accuracy_score(y_test, cat_result))
        # print("lightGBM Accuracy:", accuracy_score(y_test, gbm_result))

        average_proba = (proba1 + proba2) / 2
        soft_voting_result = np.argmax(average_proba, axis=1)
        print("Soft Voting Accuracy:", accuracy_score(y_test, soft_voting_result))

        if True:
            proba1 = xgb_model.predict_proba(test)
            proba2 = cat_model.predict_proba(test)
            # proba3 = gbm_model.predict(test, num_iteration=gbm_model.best_iteration)
            average_proba = (proba1)
            soft_voting_result = np.argmax(average_proba, axis=1)

            answer = pd.read_csv('../Database/sample_submission.csv')

            label_encoder = LabelEncoder()
            encoded_data = label_encoder.fit_transform(original['대출등급'])
            decoded_data = label_encoder.inverse_transform(soft_voting_result)

            # 복원된 데이터를 DataFrame으로 변환
            df_restored = pd.DataFrame({'대출등급': decoded_data})

            answer['대출등급']=decoded_data

            answer.to_csv('../Files/answer.csv', index=None)


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
