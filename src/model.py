import lightgbm as lgb
import numpy as np

PARAMS = dict(n_estimators=800, learning_rate=0.02, num_leaves=31, min_child_samples=15, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.5, cat_smooth=20, min_data_per_group=30, reg_lambda=1.0,
              verbose=-1)


class SeedEnsemble:
    def __init__(self, seeds=(42, 7, 2024), **params):
        self.models = [lgb.LGBMClassifier(**{**PARAMS, **params, "random_state": s}) for s in seeds]

    def fit(self, X, y):
        for m in self.models:
            m.fit(X, y)
        self.feature_names_ = list(X.columns)
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models], axis=0)

    def contributions(self, X):
        return np.mean([m.predict_proba(X, pred_contrib=True) for m in self.models], axis=0)
