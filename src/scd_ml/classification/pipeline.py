"""Classifier pipelines: a fold-local head followed by selector, balancer and model.

The head (radiomics cleaning, optional clinical metadata encoding and scaling) does not
depend on the experimental configuration, so cross-validation fits it once per fold and
reuses it. The tail (selector -> balancer -> model) is rebuilt for every configuration.
``fit_full_pipeline`` runs the exact same two steps on the full development set.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, f_classif
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier

from scd_ml.data.schemas import require_columns

from .columns import feature_columns
from .preprocessing import FoldLocalRadiomicsTransformer

SELECTIONS = ("none", "anova", "l1", "rfe")
BALANCINGS = ("none", "class_weight", "smote")
MODELS = ("logreg", "xgboost", "random_forest", "mlp", "rbf_svm")
# Models without ``class_weight``; balancing passes ``sample_weight`` at fit time.
SAMPLE_WEIGHT_MODELS = frozenset({"xgboost", "mlp"})

DEFAULT_PREPROCESSING = {
    "winsor_low": 0.01,
    "winsor_high": 0.99,
    "correlation_threshold": 0.95,
}
L1_SELECTOR_C = 0.1
RFE_STEP = 0.1

DEFAULT_MODEL_PARAMS: dict[str, dict[str, Any]] = {
    "logreg": {"C": 1.0, "max_iter": 2000},
    "xgboost": {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 6},
    "random_forest": {"n_estimators": 300, "min_samples_leaf": 1, "max_features": "sqrt"},
    "mlp": {
        "hidden_layer_sizes": [128],
        "alpha": 1e-4,
        "learning_rate_init": 1e-3,
        "batch_size": 256,
        "max_iter": 200,
    },
    "rbf_svm": {"C": 1.0, "gamma_scale": 1.0, "n_components": 1000, "max_iter": 5000},
}

_SKLEARN_VERSION = tuple(int(part) for part in sklearn.__version__.split(".")[:2])


@dataclass(frozen=True)
class ModelConfig:
    """One fully specified experimental configuration."""

    channel_set: str
    use_metadata: bool
    selection: str
    balancing: str
    model: str
    k: int | None = None
    params: dict[str, Any] = field(default_factory=dict)
    seed: int = 42
    preprocessing: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_PREPROCESSING)
    )

    def __post_init__(self) -> None:
        if self.selection not in SELECTIONS:
            raise ValueError(f"unknown selection {self.selection!r}; use {SELECTIONS}")
        if self.balancing not in BALANCINGS:
            raise ValueError(f"unknown balancing {self.balancing!r}; use {BALANCINGS}")
        if self.model not in MODELS:
            raise ValueError(f"unknown model {self.model!r}; use {MODELS}")
        if self.selection != "none" and (self.k is None or self.k < 1):
            raise ValueError(f"selection {self.selection!r} requires a positive k")

    @property
    def name(self) -> str:
        metadata = "meta" if self.use_metadata else "radiomics"
        k = f"k{self.k}" if self.selection != "none" else "kall"
        return "|".join(
            [self.channel_set, metadata, self.selection, k, self.balancing, self.model]
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> ModelConfig:
        return cls(**values)


class MetadataEncoder(TransformerMixin, BaseEstimator):
    """Median-impute numeric and one-hot encode categorical clinical metadata.

    Missing values become explicit indicators; categories unseen during ``fit`` encode
    as all zeros. Every statistic comes from the training partition and never from
    ``target``.
    """

    def __init__(
        self,
        numeric: Sequence[str] = ("age_approx",),
        categorical: Sequence[str] = ("sex", "anatom_site_1"),
    ) -> None:
        self.numeric = numeric
        self.categorical = categorical

    @staticmethod
    def _slug(value: object) -> str:
        return re.sub(r"[^0-9A-Za-z]+", "_", str(value)).strip("_").lower() or "blank"

    def fit(self, X: pd.DataFrame, y: object = None) -> MetadataEncoder:
        del y
        require_columns(X, [*self.numeric, *self.categorical], name="clinical metadata")
        self.medians_ = {}
        for column in self.numeric:
            median = pd.to_numeric(X[column], errors="coerce").median()
            self.medians_[column] = 0.0 if pd.isna(median) else float(median)
        self.categories_ = {
            column: sorted(X[column].dropna().astype(str).unique())
            for column in self.categorical
        }
        names: list[str] = []
        for column in self.numeric:
            names += [column, f"{column}__missing"]
        for column in self.categorical:
            names += [f"{column}__{self._slug(value)}" for value in self.categories_[column]]
            names.append(f"{column}__missing")
        self.feature_names_out_ = np.asarray(names, dtype=object)
        self.n_features_in_ = len(self.numeric) + len(self.categorical)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "feature_names_out_")
        require_columns(X, [*self.numeric, *self.categorical], name="clinical metadata")
        blocks: list[np.ndarray] = []
        for column in self.numeric:
            values = pd.to_numeric(X[column], errors="coerce")
            blocks.append(values.fillna(self.medians_[column]).to_numpy(dtype=float))
            blocks.append(values.isna().to_numpy(dtype=float))
        for column in self.categorical:
            values = X[column].astype("string")
            for category in self.categories_[column]:
                blocks.append(values.eq(category).fillna(False).to_numpy(dtype=float))
            blocks.append(values.isna().to_numpy(dtype=float))
        return pd.DataFrame(
            np.column_stack(blocks), columns=self.feature_names_out_, index=X.index
        )

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        del input_features
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_


class RBFFeatureMap(TransformerMixin, BaseEstimator):
    """Nystroem approximation of an RBF kernel with ``gamma = gamma_scale / n_features``.

    Tying gamma to the number of inputs (sklearn's ``gamma="scale"`` for standardized
    data) keeps one search space valid whatever ``k`` the selector keeps. An exact
    ``SVC`` is O(n^2) on ~50k rows; the approximation is linear in the rows.
    """

    def __init__(
        self, gamma_scale: float = 1.0, n_components: int = 1000, random_state: int | None = None
    ) -> None:
        self.gamma_scale = gamma_scale
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X: Any, y: object = None) -> RBFFeatureMap:
        del y
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.nystroem_ = Nystroem(
            kernel="rbf",
            gamma=self.gamma_scale / X.shape[1],
            n_components=min(int(self.n_components), X.shape[0]),
            random_state=self.random_state,
        ).fit(X)
        return self

    def transform(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "nystroem_")
        return self.nystroem_.transform(np.asarray(X, dtype=float))


def build_head(
    radiomic_columns: Sequence[str],
    metadata_columns: Sequence[str],
    preprocessing: dict[str, float],
) -> Pipeline:
    """Fold-local radiomics cleaning, optional metadata encoding and scaling."""
    transformers = [
        (
            "radiomics",
            FoldLocalRadiomicsTransformer(**preprocessing),
            list(radiomic_columns),
        )
    ]
    if metadata_columns:
        transformers.append(("metadata", MetadataEncoder(), list(metadata_columns)))
    columns = ColumnTransformer(
        transformers, remainder="drop", verbose_feature_names_out=False
    )
    return Pipeline([("columns", columns), ("scaler", StandardScaler())]).set_output(
        transform="pandas"
    )


def _logistic_l1(C: float, seed: int) -> LogisticRegression:
    options = {"C": C, "solver": "saga", "max_iter": 500, "tol": 1e-3, "random_state": seed}
    if _SKLEARN_VERSION >= (1, 8):
        return LogisticRegression(l1_ratio=1.0, **options)
    return LogisticRegression(penalty="l1", **options)


def build_selector(selection: str, k: int | None, *, n_features: int, seed: int) -> Any:
    if selection == "none":
        return "passthrough"
    k = min(int(k), n_features)
    if selection == "anova":
        return SelectKBest(f_classif, k=k)
    if selection == "l1":
        return SelectFromModel(
            _logistic_l1(L1_SELECTOR_C, seed), max_features=k, threshold=-np.inf
        )
    if selection == "rfe":
        return RFE(
            LogisticRegression(max_iter=2000, random_state=seed),
            n_features_to_select=k,
            step=RFE_STEP,
        )
    raise ValueError(f"unknown selection {selection!r}")


def build_balancer(balancing: str, *, seed: int) -> Any:
    if balancing == "smote":
        return SMOTE(random_state=seed)
    return "passthrough"


def build_model(
    name: str, params: dict[str, Any], *, balancing: str, seed: int, n_jobs: int = 1
) -> Any:
    params = {**DEFAULT_MODEL_PARAMS[name], **params}
    class_weight = "balanced" if balancing == "class_weight" else None
    if name == "logreg":
        return LogisticRegression(class_weight=class_weight, random_state=seed, **params)
    if name == "xgboost":
        return XGBClassifier(
            objective="multi:softprob",
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=n_jobs,
            verbosity=0,
            **params,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            class_weight=class_weight, random_state=seed, n_jobs=n_jobs, **params
        )
    if name == "mlp":
        params = {**params, "hidden_layer_sizes": tuple(params["hidden_layer_sizes"])}
        return MLPClassifier(early_stopping=True, random_state=seed, **params)
    if name == "rbf_svm":
        feature_map = RBFFeatureMap(
            gamma_scale=params.pop("gamma_scale"),
            n_components=params.pop("n_components"),
            random_state=seed,
        )
        svm = LinearSVC(class_weight=class_weight, random_state=seed, **params)
        kernel_svm = Pipeline([("rbf", feature_map), ("svm", svm)])
        return CalibratedClassifierCV(kernel_svm, cv=3, method="sigmoid", n_jobs=n_jobs)
    raise ValueError(f"unknown model {name!r}")


def build_tail(config: ModelConfig, *, n_features: int, n_jobs: int = 1) -> ImbPipeline:
    return ImbPipeline(
        [
            (
                "selector",
                build_selector(
                    config.selection, config.k, n_features=n_features, seed=config.seed
                ),
            ),
            ("balancer", build_balancer(config.balancing, seed=config.seed)),
            (
                "model",
                build_model(
                    config.model,
                    config.params,
                    balancing=config.balancing,
                    seed=config.seed,
                    n_jobs=n_jobs,
                ),
            ),
        ]
    )


def fit_params(config: ModelConfig, y: np.ndarray) -> dict[str, np.ndarray]:
    if config.balancing == "class_weight" and config.model in SAMPLE_WEIGHT_MODELS:
        return {"model__sample_weight": compute_sample_weight("balanced", y)}
    return {}


def fit_tail(
    config: ModelConfig, X: pd.DataFrame, y: np.ndarray, *, n_jobs: int = 1
) -> ImbPipeline:
    tail = build_tail(config, n_features=X.shape[1], n_jobs=n_jobs)
    return tail.fit(X, y, **fit_params(config, y))


def model_input_columns(frame_columns: Sequence[str], config: ModelConfig) -> list[str]:
    radiomics, metadata = feature_columns(
        frame_columns, config.channel_set, use_metadata=config.use_metadata
    )
    return [*radiomics, *metadata]


def fit_full_pipeline(
    config: ModelConfig, frame: pd.DataFrame, y: np.ndarray, *, n_jobs: int = 1
) -> ImbPipeline:
    """Fit head and tail on ``frame`` and return them as one predict-ready pipeline."""
    radiomics, metadata = feature_columns(
        frame.columns, config.channel_set, use_metadata=config.use_metadata
    )
    head = build_head(radiomics, metadata, config.preprocessing)
    transformed = head.fit_transform(frame[[*radiomics, *metadata]])
    tail = fit_tail(config, transformed, y, n_jobs=n_jobs)
    return ImbPipeline([("head", head), *tail.steps])


def selected_feature_names(tail: ImbPipeline, input_features: Sequence[str]) -> list[str]:
    """Names kept by a fitted tail's selector (all inputs when there is none)."""
    selector = tail.named_steps["selector"]
    if selector == "passthrough" or selector is None:
        return list(input_features)
    return [str(name) for name in np.asarray(input_features)[selector.get_support()]]
