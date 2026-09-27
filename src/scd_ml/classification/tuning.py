"""Optuna search spaces and the grouped-CV objective for stage-2 tuning."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import optuna

from .cv import FoldData, evaluate_fold
from .metrics import PRIMARY_METRIC
from .pipeline import ModelConfig

K_MIN = 10
K_STEP = 5


def suggest_model_params(trial: optuna.Trial, model: str) -> dict[str, Any]:
    """Sample model hyperparameters; returned values are JSON-serialisable."""
    if model == "logreg":
        return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True), "max_iter": 2000}
    if model == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    if model == "random_forest":
        max_depth = trial.suggest_categorical("max_depth", [0, 10, 20, 40])
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": max_depth or None,
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20, log=True),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3]),
        }
    if model == "mlp":
        n_layers = trial.suggest_int("n_layers", 1, 2)
        first = trial.suggest_int("units_0", 64, 512, log=True)
        layers = [first] if n_layers == 1 else [first, trial.suggest_int("units_1", 32, 256)]
        return {
            "hidden_layer_sizes": layers,
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [256, 512]),
            "max_iter": 200,
        }
    if model == "rbf_svm":
        return {
            "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
            "gamma_scale": trial.suggest_float("gamma_scale", 0.05, 20.0, log=True),
            "n_components": trial.suggest_categorical("n_components", [500, 1000, 2000]),
            "max_iter": 5000,
        }
    raise ValueError(f"unknown model {model!r}")


def suggest_config(
    trial: optuna.Trial, base: ModelConfig, *, n_features: int, k_max: int
) -> ModelConfig:
    """Sample model hyperparameters and, when a selector is active, its ``k``."""
    k = None
    if base.selection != "none":
        upper = max(K_MIN, min(k_max, n_features))
        k = trial.suggest_int("k", K_MIN, upper, step=K_STEP)
    return replace(base, params=suggest_model_params(trial, base.model), k=k)


def make_objective(
    base: ModelConfig,
    folds: Sequence[FoldData],
    *,
    k_max: int,
    model_jobs: int = 1,
) -> Callable[[optuna.Trial], float]:
    """Mean validation F1-macro over folds, reported per fold for pruning."""
    n_features = min(data.X_train.shape[1] for data in folds)

    def objective(trial: optuna.Trial) -> float:
        config = suggest_config(trial, base, n_features=n_features, k_max=k_max)
        trial.set_user_attr("config", config.to_dict())
        scores: list[float] = []
        for step, data in enumerate(folds):
            result = evaluate_fold(config, data, n_jobs=model_jobs)
            scores.append(result.metrics[PRIMARY_METRIC])
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("fold_scores", scores)
        return float(np.mean(scores))

    return objective


def create_study(
    name: str,
    storage: str,
    *,
    seed: int,
    n_startup_trials: int,
    n_warmup_steps: int,
) -> optuna.Study:
    return optuna.create_study(
        study_name=name,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=n_startup_trials, n_warmup_steps=n_warmup_steps
        ),
        load_if_exists=True,
    )


def finished_trials(study: optuna.Study) -> int:
    states = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    return len(study.get_trials(deepcopy=False, states=states))
