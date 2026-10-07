import numpy as np
from scipy.sparse import issparse

class SimpleCountMultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes_ = None
        self.log_prior_ = None
        self.log_prob_ = None

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        class_count = np.bincount(y_idx)
        self.log_prior_ = np.log(class_count) - np.log(len(y))

        word_count = np.zeros((n_classes, n_features), dtype=float)
        for c in range(n_classes):
            rows = (y_idx == c)
            if issparse(X):
                word_count[c] = np.asarray(X[rows].sum(axis=0)).ravel()
            else:
                word_count[c] = X[rows].sum(axis=0)

        smoothed = word_count + self.alpha
        denom = smoothed.sum(axis=1, keepdims=True)
        self.log_prob_ = np.log(smoothed / denom)
        return self

    def predict(self, X):
        if issparse(X):
            scores = X @ self.log_prob_.T
        else:
            scores = np.dot(X, self.log_prob_.T)
        scores = scores + self.log_prior_
        return self.classes_[scores.argmax(axis=1)]
