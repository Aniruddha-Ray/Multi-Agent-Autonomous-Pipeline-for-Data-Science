"""Training Agent.

Extracted verbatim from Notebook Cell 9 ("CELL 6e — TRAINING AGENT PART 1")
and Cell 10 ("CELL 6f — TRAINING AGENT PART 2"), merged into one module —
the two-cell split was notebook-length driven, not a logical seam (per the
earlier modularization analysis). Fully deterministic — no LLM call
anywhere in this agent.
"""
from __future__ import annotations

from typing import Any, Optional

import catboost as cb
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
    precision_score, r2_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
import sys
import os
from pathlib import Path 

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.config.settings import CFG, Config
from src.models.state import PipelineState

DEFAULT_HYPERPARAMS: dict[str, dict[str, Any]] = {
    "Random Forest":       {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1},
    "XGBoost":              {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1},
    "LightGBM":              {"n_estimators": 200, "max_depth": -1, "learning_rate": 0.1},
    "CatBoost":              {"iterations": 200, "depth": 6, "learning_rate": 0.1},
    "Logistic Regression": {"C": 1.0},
    "Linear Regression":   {},
    "ElasticNet":            {"alpha": 1.0, "l1_ratio": 0.5},
    "SVM":                    {"C": 1.0, "kernel": "rbf"},
}

TOP_N_MODELS_TO_TUNE: int = 3


def build_estimator(
    model_name: str,
    problem_type: str,
    params: Optional[dict[str, Any]] = None,
    handle_imbalance: bool = False,
) -> Any:
    merged_params = {**DEFAULT_HYPERPARAMS.get(model_name, {}), **(params or {})}
    is_classification = problem_type == "classification"

    if is_classification and handle_imbalance and model_name in (
        "Random Forest", "Logistic Regression", "SVM", "LightGBM"
    ):
        merged_params.setdefault("class_weight", "balanced")

    if model_name == "Random Forest":
        cls = RandomForestClassifier if is_classification else RandomForestRegressor
        return cls(**merged_params, random_state=CFG.random_state)
    if model_name == "XGBoost":
        cls = xgb.XGBClassifier if is_classification else xgb.XGBRegressor
        extra = {"eval_metric": "logloss"} if is_classification else {}
        return cls(**merged_params, **extra, random_state=CFG.random_state, verbosity=0)
    if model_name == "LightGBM":
        cls = lgb.LGBMClassifier if is_classification else lgb.LGBMRegressor
        return cls(**merged_params, random_state=CFG.random_state, verbose=-1)
    if model_name == "CatBoost":
        cls = cb.CatBoostClassifier if is_classification else cb.CatBoostRegressor
        return cls(**merged_params, random_state=CFG.random_state, verbose=0)
    if model_name == "Logistic Regression":
        return LogisticRegression(**merged_params, random_state=CFG.random_state, max_iter=1000)
    if model_name == "Linear Regression":
        return LinearRegression(**merged_params)
    if model_name == "ElasticNet":
        return ElasticNet(**merged_params, random_state=CFG.random_state)
    if model_name == "SVM":
        cls = SVC if is_classification else SVR
        extra = {"probability": True} if is_classification else {}
        return cls(**merged_params, **extra)

    raise ValueError(f"Unknown model_name: '{model_name}'")


def build_full_pipeline(
    preprocessor: ColumnTransformer,
    decision: dict[str, Any],
    problem_type: str,
    estimator: Any,
) -> Pipeline:
    steps: list[tuple[str, Any]] = [("preprocess", clone(preprocessor))]

    if decision.get("feature_selection_k"):
        score_func = f_classif if problem_type == "classification" else f_regression
        steps.append(("select", SelectKBest(score_func=score_func, k=decision["feature_selection_k"])))

    if decision.get("use_pca"):
        n_components = decision.get("pca_n_components") or CFG.pca_variance_threshold
        steps.append(("pca", PCA(n_components=n_components, random_state=CFG.random_state)))

    steps.append(("model", estimator))
    return Pipeline(steps)


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray], problem_type: str
) -> dict[str, float]:
    if problem_type == "classification":
        metrics: dict[str, float] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }
        if y_proba is not None:
            try:
                if y_proba.shape[1] == 2:
                    metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
                else:
                    metrics["roc_auc"] = float(
                        roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
                    )
            except ValueError:
                pass
        return metrics

    mse = mean_squared_error(y_true, y_pred)
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def get_primary_scorer(problem_type: str) -> str:
    return "f1_weighted" if problem_type == "classification" else "neg_root_mean_squared_error"


def cross_val_primary_score(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, problem_type: str, cfg: Config
) -> float:
    cv = (
        StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state)
        if problem_type == "classification"
        else KFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state)
    )
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=get_primary_scorer(problem_type), n_jobs=1)
    return float(np.mean(scores))


def _suggest_hyperparams(trial: "optuna.Trial", model_name: str) -> dict[str, Any]:
    if model_name == "Random Forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }
    if model_name == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
    if model_name == "LightGBM":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        }
    if model_name == "CatBoost":
        return {
            "iterations": trial.suggest_int("iterations", 100, 500),
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        }
    if model_name == "Logistic Regression":
        return {"C": trial.suggest_float("C", 1e-3, 10.0, log=True)}
    if model_name == "Linear Regression":
        return {}
    if model_name == "ElasticNet":
        return {
            "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        }
    if model_name == "SVM":
        return {
            "C": trial.suggest_float("C", 1e-2, 10.0, log=True),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        }
    return {}


def tune_model_with_optuna(
    model_name: str,
    problem_type: str,
    preprocessor: ColumnTransformer,
    decision: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    cfg: Config,
) -> dict[str, Any]:
    handle_imbalance = bool(decision.get("handle_imbalance", False))

    if model_name == "Linear Regression":
        estimator = build_estimator(model_name, problem_type, handle_imbalance=handle_imbalance)
        pipeline = build_full_pipeline(preprocessor, decision, problem_type, estimator)
        score = cross_val_primary_score(pipeline, X, y, problem_type, cfg)
        return {"best_params": {}, "best_score": score, "n_trials": 1}

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_hyperparams(trial, model_name)
        estimator = build_estimator(model_name, problem_type, params, handle_imbalance=handle_imbalance)
        pipeline = build_full_pipeline(preprocessor, decision, problem_type, estimator)
        try:
            return cross_val_primary_score(pipeline, X, y, problem_type, cfg)
        except Exception as exc:  # noqa: BLE001
            raise optuna.TrialPruned(str(exc))

    sampler = optuna.samplers.TPESampler(seed=cfg.random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=cfg.n_optuna_trials, show_progress_bar=False)

    return {"best_params": study.best_params, "best_score": study.best_value, "n_trials": len(study.trials)}


def training_agent_node(state: PipelineState) -> PipelineState:
    metadata = state["metadata"]
    problem_type = metadata["problem_type"]
    decision = state["features"]["decision"]
    preprocessor = state["features"]["preprocessor"]
    target_column = state["target_column"]
    df = state["dataset"]

    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CFG.test_size, random_state=CFG.random_state,
        stratify=y if problem_type == "classification" else None,
    )

    candidates = state["model_recommendation"]["ranked_models"][:TOP_N_MODELS_TO_TUNE]

    trained_models: dict[str, Any] = {}
    tuning_results: dict[str, Any] = {}
    test_metrics: dict[str, dict[str, float]] = {}

    for model_name in candidates:
        tuning = tune_model_with_optuna(model_name, problem_type, preprocessor, decision, X_train, y_train, CFG)
        tuning_results[model_name] = tuning

        best_estimator = build_estimator(
            model_name, problem_type, tuning["best_params"],
            handle_imbalance=bool(decision.get("handle_imbalance", False)),
        )
        best_pipeline = build_full_pipeline(preprocessor, decision, problem_type, best_estimator)
        best_pipeline.fit(X_train, y_train)

        y_pred = best_pipeline.predict(X_test)
        y_proba = best_pipeline.predict_proba(X_test) if hasattr(best_pipeline, "predict_proba") else None
        metrics = compute_metrics(y_test.values, y_pred, y_proba, problem_type)
        test_metrics[model_name] = metrics
        trained_models[model_name] = best_pipeline

        headline = metrics.get("f1", metrics.get("rmse"))
        print(
            f"  [{model_name}] CV best={tuning['best_score']:.4f} ({tuning['n_trials']} trials) "
            f"| held-out {'f1' if problem_type == 'classification' else 'rmse'}={headline:.4f}"
        )

    def _rank_key(name: str) -> float:
        m = test_metrics[name]
        return m["f1"] if problem_type == "classification" else -m["rmse"]

    best_model_name = max(test_metrics, key=_rank_key)

    state["models"] = {
        "candidates_evaluated": candidates,
        "trained_pipelines": trained_models,
        "tuning_results": tuning_results,
        "best_model_name": best_model_name,
        "train_test_split": {"test_size": CFG.test_size, "n_train": len(X_train), "n_test": len(X_test)},
        "X_test": X_test,
        "y_test": y_test,
    }
    state["metrics"] = {
        "per_model_test_metrics": test_metrics,
        "best_model_name": best_model_name,
        "best_model_metrics": test_metrics[best_model_name],
        "primary_scorer": get_primary_scorer(problem_type),
    }
    state.setdefault("history", []).append({"node": "training_agent", "status": "ok", "best_model": best_model_name})
    return state