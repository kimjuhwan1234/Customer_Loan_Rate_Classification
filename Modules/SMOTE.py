import numpy as np
import pandas as pd
from collections import Counter

class SMOTE:
    def __init__(self, ratio, K, features):
        self.ratio = ratio
        self.K = K
        self.features = features

    def nearest_neighbour(self, X, x, K, features):
        X_selected = X[:, features]
        x_selected = x[features]

        distances = np.linalg.norm(X_selected - x_selected, axis=1)
        indices = np.argsort(distances)[1:K + 1]

        differences = X_selected[indices] - x_selected
        weights = np.random.random((K, 1))
        additive = differences * weights
        return additive

    def generate_new_samples(self, row, X, K, features):
        row = row.values
        additive = self.nearest_neighbour(X, row, K, features)
        new_samples = np.tile(row, (K, 1))
        new_samples[:, features] += additive
        return new_samples

    def integrate_result(self, X, y, K, features):
        X_df = pd.DataFrame(X)
        minority_class = min(Counter(y), key=Counter(y).get)
        X_minority = X_df[y == minority_class]

        new_samples_list = X_minority.apply(lambda row: self.generate_new_samples(row, X, K, features), axis=1)
        new_samples = np.vstack(new_samples_list)

        new_X = np.vstack([X, new_samples])
        new_y = np.hstack([y, np.array([minority_class] * len(new_samples))])

        return new_X, new_y

    def SMOTE(self, X, y):
        if self.features is None:
            self.features = list(range(X.shape[1]))

        self.X = X
        self.y = y

        class_counts = Counter(self.y)
        minority_class = min(class_counts, key=class_counts.get)

        while class_counts[minority_class] / len(self.y) < self.ratio:
            X, y = self.integrate_result(self.X, self.y, self.K, self.features)
            class_counts = Counter(y)
            minority_class = min(class_counts, key=class_counts.get)

        return X, y

if __name__ == "__main__":
    test_in = pd.read_csv('../Database/test_pre_input.csv', index_col=0)
    test_out = pd.read_csv('../Database/test_pre_output.csv', index_col=0)
    # SMOTE 적용
    features = [0, 1, 2, 3, 4, 5, 6, 7, 11, 12]
    S = SMOTE(0.1, 2, features)
    X_balanced, y_balanced = S.SMOTE(test_in, test_out)

    print("Original class distribution:", Counter(test_out))
    print("New class distribution:", Counter(y_balanced))