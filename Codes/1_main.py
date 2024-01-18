from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

import pandas as pd


if __name__ == "__main__":
    train = pd.read_csv('../Database/train_modified.csv', index_col='ID')
    X = train.drop(columns=['대출등급'])
    y = train['대출등급']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestClassifier(n_estimators=10, max_depth=None, random_state=42)
    gb_model = GradientBoostingClassifier(n_estimators=10, random_state=42)
    xgb_model = XGBClassifier(n_estimators=1000, random_state=2024, learning_rate=0.01, max_depth=10)
    cat_model = CatBoostClassifier(random_state=2024,
                            n_estimators=1000,
                            learning_rate=0.01,
                            depth=10,
                            l2_leaf_reg=3,
                            metric_period=1000)

    voting_clf = VotingClassifier(estimators=[
        ('random_forest', rf_model),
        ('gradient_boosting', gb_model),
        ('XGB_Boosting', xgb_model),
        ('CatBoosting', cat_model)
    ], voting='soft')

    voting_clf.fit(X_train, y_train)

    # 각 모델과 Voting Classifier의 정확도 출력
    print("Voting Classifier Accuracy:", accuracy_score(y_test, voting_clf.predict(X_test)))
