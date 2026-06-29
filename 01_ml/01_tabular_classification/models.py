from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from _runtime import (
    ModelResult,
    SEED,
    binary_metrics,
    process,
    timed_fit_predict,
    to_dense_float32,
    train_torch_classifier,
)
from dataset import AdultClassificationSplit

MODEL_NAMES = ['dummy_prior', 'logistic_regression', 'random_forest', 'gpu_mlp']


def build_sklearn_models(preprocessor: Any) -> dict[str, Any]:
    """Adult income 분류에서 비교할 CPU/sklearn 후보 모델을 만든다."""
    return {
        'dummy_prior': DummyClassifier(strategy='prior'),
        'logistic_regression': Pipeline([
            ('preprocessor', clone(preprocessor)),
            ('model', LogisticRegression(max_iter=1200, class_weight='balanced', n_jobs=-1)),
        ]),
        'random_forest': Pipeline([
            ('preprocessor', clone(preprocessor)),
            ('model', RandomForestClassifier(
                n_estimators=260,
                min_samples_leaf=2,
                class_weight='balanced_subsample',
                n_jobs=-1,
                random_state=SEED,
            )),
        ]),
    }


def fit_sklearn_model_suite(
    models: dict[str, Any],
    split: AdultClassificationSplit,
) -> dict[str, ModelResult]:
    """같은 train/test split에서 baseline과 sklearn 모델을 같은 지표 구조로 평가한다."""
    results: dict[str, ModelResult] = {}
    for name, model in models.items():
        model, y_pred, y_score, fit_time, predict_time, peak_rss = timed_fit_predict(
            model,
            split.X_train,
            split.y_train,
            split.X_test,
        )
        if y_score is None:
            y_score = y_pred.astype(float)
        results[name] = ModelResult(
            name=name,
            metrics=binary_metrics(split.y_test, y_pred, np.asarray(y_score)),
            fit_time_sec=fit_time,
            predict_time_sec=predict_time,
            peak_rss_mb=peak_rss,
            y_pred=np.asarray(y_pred),
            y_score=np.asarray(y_score),
        )
    return results


def train_mlp_candidate(split: AdultClassificationSplit, device: str) -> ModelResult:
    """표 데이터를 dense tensor로 바꿔 작은 MLP 후보를 학습하고 sklearn 후보와 비교한다."""
    mlp_transformer = clone(split.mlp_preprocessor)
    X_train_mlp = to_dense_float32(mlp_transformer.fit_transform(split.X_train))
    X_valid_mlp = to_dense_float32(mlp_transformer.transform(split.X_valid))
    X_test_mlp = to_dense_float32(mlp_transformer.transform(split.X_test))

    rss_before = process.memory_info().rss
    t0 = time.perf_counter()
    y_pred_mlp, y_prob_mlp, extras = train_torch_classifier(
        X_train_mlp,
        split.y_train,
        X_valid_mlp,
        split.y_valid,
        X_test_mlp,
        n_classes=2,
        device=device,
        epochs=14,
        batch_size=768,
    )
    fit_time = time.perf_counter() - t0
    peak_rss = max(rss_before, process.memory_info().rss) / (1024 ** 2)
    return ModelResult(
        name='gpu_mlp',
        metrics=binary_metrics(split.y_test, y_pred_mlp, y_prob_mlp[:, 1]),
        fit_time_sec=fit_time,
        predict_time_sec=0.0,
        peak_rss_mb=peak_rss,
        y_pred=y_pred_mlp,
        y_score=y_prob_mlp[:, 1],
        extras=extras,
    )


def choose_best_model(results: dict[str, ModelResult], primary_metric: str) -> tuple[str, ModelResult]:
    """단원이 정한 primary metric으로 대표 모델을 고른다."""
    best_name = max(results, key=lambda model_name: results[model_name].metrics[primary_metric])
    return best_name, results[best_name]


def fit_analysis_pipeline(
    results: dict[str, ModelResult],
    sklearn_models: dict[str, Any],
    split: AdultClassificationSplit,
    primary_metric: str,
) -> tuple[str, Any]:
    """Permutation importance처럼 sklearn pipeline이 필요한 분석용 모델을 준비한다."""
    sklearn_result_names = [name for name in results if name in sklearn_models]
    analysis_name = max(sklearn_result_names, key=lambda name: results[name].metrics[primary_metric])
    analysis_pipeline = sklearn_models[analysis_name].fit(split.X_train, split.y_train)
    return analysis_name, analysis_pipeline
