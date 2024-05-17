import optuna
import warnings
import pandas as pd
from Modules.Esemble import *
from imblearn.over_sampling import SMOTE, ADASYN, KMeansSMOTE
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    train = pd.read_csv('Database/train_pre.csv', index_col=0)
    X = train.drop(columns=['대출등급'])
    y = train['대출등급']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # smote = SMOTE(random_state=42)
    # X_res, y_res = smote.fit_resample(X_train, y_train)

    method = 0  # {DT=0, lightGBM=1, XGBoost=2, CatBoost=3}
    E = Esemble(method, X_train, X_val, y_train, y_val, 1000, 'None')

    study = optuna.create_study(direction='maximize')
    study.optimize(E.objective, n_trials=100)

    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)

    E.save_best_model(study.best_trial.params)
