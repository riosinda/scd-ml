"""Fold-local preprocessing for numeric radiomic features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class FoldLocalRadiomicsTransformer(TransformerMixin, BaseEstimator):
    """Impute, winsorize and prune correlated radiomic features.

    The transformer deliberately accepts and returns pandas data frames so feature
    names remain auditable. Call ``fit`` only on the training partition and call
    ``transform`` on both training and validation/test partitions.
    """

    def __init__(
        self,
        *,
        winsor_low: float = 0.01,
        winsor_high: float = 0.99,
        correlation_threshold: float = 0.95,
    ) -> None:
        self.winsor_low = winsor_low
        self.winsor_high = winsor_high
        self.correlation_threshold = correlation_threshold

    @staticmethod
    def _numeric_frame(X: pd.DataFrame, *, name: str) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame")
        if X.columns.duplicated().any():
            duplicates = X.columns[X.columns.duplicated()].astype(str).tolist()[:5]
            raise ValueError(f"{name} has duplicated columns: {duplicates}")
        if not all(isinstance(column, str) for column in X.columns):
            raise ValueError(f"{name} feature names must be strings")

        converted = X.apply(pd.to_numeric, errors="coerce")
        invalid = X.notna() & converted.isna()
        if invalid.any().any():
            column = invalid.any(axis=0).idxmax()
            raise ValueError(f"{name}.{column} contains non-numeric values")
        return converted.replace([np.inf, -np.inf], np.nan)

    def _validate_parameters(self) -> None:
        if not 0 <= self.winsor_low < self.winsor_high <= 1:
            raise ValueError("winsor bounds must satisfy 0 <= low < high <= 1")
        if not 0 < self.correlation_threshold <= 1:
            raise ValueError("correlation_threshold must be in (0, 1]")

    def fit(self, X: pd.DataFrame, y: object = None) -> "FoldLocalRadiomicsTransformer":
        """Learn every preprocessing statistic from ``X`` only."""
        del y
        self._validate_parameters()
        numeric = self._numeric_frame(X, name="radiomic training frame")
        if numeric.empty or numeric.shape[1] == 0:
            raise ValueError("radiomic training frame must contain rows and features")

        self.feature_names_in_ = np.asarray(numeric.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        self.all_missing_features_ = numeric.columns[numeric.isna().all()].tolist()

        usable = numeric.drop(columns=self.all_missing_features_)
        if usable.shape[1] == 0:
            raise ValueError("all radiomic features are missing in the training partition")

        self.medians_ = usable.median(axis=0)
        imputed = usable.fillna(self.medians_)
        self.lower_bounds_ = imputed.quantile(self.winsor_low)
        self.upper_bounds_ = imputed.quantile(self.winsor_high)
        winsorized = imputed.clip(
            lower=self.lower_bounds_, upper=self.upper_bounds_, axis=1
        )

        correlation = winsorized.corr().abs()
        active_features = list(winsorized.columns)
        correlation_drops: list[dict[str, object]] = []

        while True:
            active_correlation = correlation.loc[active_features, active_features]
            upper = active_correlation.where(
                np.triu(np.ones(active_correlation.shape), k=1).astype(bool)
            )
            high_pairs = upper.stack()[lambda values: values > self.correlation_threshold]
            if high_pairs.empty:
                break

            involved = sorted(
                set(high_pairs.index.get_level_values(0))
                | set(high_pairs.index.get_level_values(1))
            )
            redundancy = active_correlation.loc[involved, involved].mean(axis=1)
            maximum = redundancy.max()
            loser = sorted(redundancy[redundancy == maximum].index)[0]
            related = high_pairs[
                (high_pairs.index.get_level_values(0) == loser)
                | (high_pairs.index.get_level_values(1) == loser)
            ]
            correlation_drops.append(
                {
                    "dropped_feature": loser,
                    "reason": f"corr>{self.correlation_threshold}",
                    "max_abs_correlation": float(related.max()),
                    "n_high_corr_pairs_at_removal": int(len(related)),
                }
            )
            active_features.remove(loser)

        missing_drops = [
            {
                "dropped_feature": feature,
                "reason": "all_missing_in_fold_train",
                "max_abs_correlation": np.nan,
                "n_high_corr_pairs_at_removal": np.nan,
            }
            for feature in self.all_missing_features_
        ]
        self.selected_features_ = active_features
        self.dropped_features_ = pd.DataFrame(
            [*missing_drops, *correlation_drops],
            columns=[
                "dropped_feature",
                "reason",
                "max_abs_correlation",
                "n_high_corr_pairs_at_removal",
            ],
        )
        self.feature_parameters_ = pd.DataFrame(
            {
                "feature": usable.columns,
                "median": self.medians_.reindex(usable.columns).to_numpy(),
                "lower_bound": self.lower_bounds_.reindex(usable.columns).to_numpy(),
                "upper_bound": self.upper_bounds_.reindex(usable.columns).to_numpy(),
                "selected": usable.columns.isin(self.selected_features_),
            }
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply training-fold statistics without learning from ``X``."""
        check_is_fitted(self, "selected_features_")
        numeric = self._numeric_frame(X, name="radiomic transform frame")
        missing = sorted(set(self.feature_names_in_) - set(numeric.columns))
        if missing:
            raise ValueError(f"radiomic transform frame is missing features: {missing[:5]}")

        usable_features = self.medians_.index.tolist()
        transformed = numeric.loc[:, usable_features].fillna(self.medians_)
        transformed = transformed.clip(
            lower=self.lower_bounds_, upper=self.upper_bounds_, axis=1
        )
        transformed = transformed.loc[:, self.selected_features_]
        if transformed.isna().any().any():
            raise RuntimeError("fold-local preprocessing left missing radiomic values")
        return transformed

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        del input_features
        check_is_fitted(self, "selected_features_")
        return np.asarray(self.selected_features_, dtype=object)
