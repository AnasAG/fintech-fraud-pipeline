"""
Custom feature encoders for the IEEE-CIS dataset.

Two key encoders here:

1. TargetEncoder
   Replaces a categorical value with the mean fraud rate of that category.
   Example: email domain "gmail.com" → 0.032 (3.2% fraud rate)

   Why not one-hot encoding?
   Some columns like DeviceInfo have 1,000+ unique values. One-hot explodes
   the feature space and creates sparse features that hurt tree models.
   Target encoding collapses each value to a single float.

   Cross-validation fold approach:
   Naive target encoding leaks the label into the feature. If you encode on the
   full training set, the model sees the fraud label encoded in the feature it's
   predicting — a form of target leakage. We use k-fold CV to compute the encoding
   on held-out folds, preventing this.

2. FrequencyEncoder
   Replaces a categorical value with how often it appears in the dataset.
   Example: card1 value 9500 → 234 (appears 234 times)

   Useful for device/card identifiers where frequency signals 'is this a normal
   device or one we've never seen?' — rare values correlate with fraud.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold


class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Mean target encoding with cross-validation leak prevention.

    Fit stores category→mean_target mapping.
    Transform replaces categories with their mean target from training.
    Unseen categories at inference time get the global mean (smoothing fallback).

    Parameters
    ----------
    cols : list of str
        Columns to encode.
    n_splits : int
        Number of CV folds for leak-free training encoding.
    smoothing : float
        Blend factor toward global mean for rare categories.
        Higher = more regularisation for rare categories.
    """

    def __init__(self, cols: list[str], n_splits: int = 5, smoothing: float = 1.0):
        self.cols = cols
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.global_mean_: float = 0.0
        self.encoding_map_: dict[str, dict] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TargetEncoder":
        self.global_mean_ = y.mean()
        for col in self.cols:
            # Per-category mean target with smoothing
            stats = y.groupby(X[col]).agg(["mean", "count"])
            # Smoothed mean: blend category mean toward global mean for small groups
            smooth = (stats["count"] * stats["mean"] + self.smoothing * self.global_mean_) / (
                stats["count"] + self.smoothing
            )
            self.encoding_map_[col] = smooth.to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.cols:
            X[col] = X[col].map(self.encoding_map_[col]).fillna(self.global_mean_)
        return X

    def fit_transform_train(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        CV-based encoding for training data (leak-free).

        Each row is encoded using a mapping built on the OTHER folds,
        so the encoded value never contains information from that row's own label.
        """
        self.fit(X, y)  # fit global map for inference
        X_enc = X.copy()
        kf = KFold(n_splits=self.n_splits, shuffle=False)

        for col in self.cols:
            encoded = np.full(len(X), self.global_mean_)
            for train_idx, val_idx in kf.split(X):
                fold_map = y.iloc[train_idx].groupby(X[col].iloc[train_idx]).mean()
                encoded[val_idx] = X[col].iloc[val_idx].map(fold_map).fillna(self.global_mean_)
            X_enc[col] = encoded

        return X_enc


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Replace category values with their frequency in the training set.

    Why this works for fraud:
    A card number that appears 5,000 times is a normal recurring card.
    A card number that appears once is a new or throwaway card — higher fraud risk.
    Frequency directly encodes this signal without requiring any label information.

    Parameters
    ----------
    cols : list of str
        Columns to encode.
    normalize : bool
        If True, encode as fraction of total rows. If False, raw count.
    """

    def __init__(self, cols: list[str], normalize: bool = True):
        self.cols = cols
        self.normalize = normalize
        self.freq_map_: dict[str, dict] = {}

    def fit(self, X: pd.DataFrame, y=None) -> "FrequencyEncoder":
        for col in self.cols:
            freq = X[col].value_counts(normalize=self.normalize)
            self.freq_map_[col] = freq.to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.cols:
            # Unseen values at inference → 0.0 (never seen = rare = suspicious)
            X[col] = X[col].map(self.freq_map_[col]).fillna(0.0)
        return X
