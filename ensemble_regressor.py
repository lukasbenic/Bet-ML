import numpy as np


class EnsembleRegressor:
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights
        if weights is not None:
            if len(weights) != len(models):
                raise ValueError("Number of weights must match the number of models.")
            total_weight = sum(weights)
            if total_weight == 0:
                raise ValueError("The sum of weights must be greater than 0.")
            self.weights = [w / total_weight for w in weights]

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)

    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models])
        if self.weights is not None:
            return np.average(predictions, axis=1, weights=self.weights)
        else:
            return np.mean(predictions, axis=1)
