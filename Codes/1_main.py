from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

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
        rf_model = RandomForestClassifier()
        param_grid = {
            'n_estimators': [10, 50, 100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2']
        }
        grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        print("Best Hyperparameters:", grid_search.best_params_)

        # rf_model = RandomForestClassifier(n_estimators=10, max_depth=None, random_state=42)
        # rf_model.fit(X_train, y_train)
        # print("Random Forest Accuracy:", accuracy_score(y_test, rf_model.predict(X_test)))

    GBM = True
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
        xgb_model = XGBClassifier(n_estimators=1000, random_state=2024, learning_rate=0.01, max_depth=10, **params)

        xgb_model.fit(X_train, y_train)
        print("XGBoost Accuracy:", accuracy_score(y_test, xgb_model.predict(X_test)))


    Cat = False
    if Cat:
        cat_model = CatBoostClassifier(random_state=2024,
                                       n_estimators=1000,
                                       learning_rate=0.01,
                                       depth=10,
                                       l2_leaf_reg=3,
                                       metric_period=1000,
                                       task_type='GPU')
        cat_model.fit(X_train, y_train)
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
