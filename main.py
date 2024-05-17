import optuna
import warnings
import pandas as pd
from Modules.Esemble import *
from imblearn.over_sampling import SMOTE, ADASYN, KMeansSMOTE

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    X_train = pd.read_csv('Database/X_train.csv', index_col=0)
    y_train = pd.read_csv('Database/y_train.csv', index_col=0)
    X_val = pd.read_csv('Database/X_val.csv', index_col=0)
    y_val = pd.read_csv('Database/y_val.csv', index_col=0)

    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # adasyn = ADASYN(random_state=42)
    # X_train, y_train = adasyn.fit_resample(X_train, y_train)
    #
    # kmsmote = KMeansSMOTE(random_state=42)
    # X_train, y_train = kmsmote.fit_resample(X_train, y_train)

    for i in range(4):
        # {DT=0, lightGBM=1, XGBoost=2, CatBoost=3}
        E = Esemble(i, X_train, X_val, y_train, y_val, 1000, 'SMOTE')

        study = optuna.create_study(direction='maximize')
        study.optimize(E.objective, n_trials=100)

        print('Number of finished trials:', len(study.trials))
        print('Best trial:', study.best_trial.params)
        E.save_best_model(study.best_trial.params)
