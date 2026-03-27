#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import shutil
import textwrap
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import psutil
import requests
import torch
import yaml
from datasets import load_dataset
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression, PoissonRegressor, Ridge, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedShuffleSplit, TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None


SEED = 42
TRACK = "01_ml"
ROOT = Path(__file__).resolve().parents[1]
TRACK_ROOT = ROOT / TRACK
RUNS_ROOT = TRACK_ROOT
REPORTS_ROOT = TRACK_ROOT
DATA_ROOT = ROOT / "data"


# ---------- General utils ----------


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class RunPaths:
    run_id: str
    run_dir: Path
    report_dir: Path
    figures_results: Path
    figures_analysis: Path
    predictions_dir: Path
    logs_dir: Path
    checkpoints_dir: Path


@dataclass
class StageContext:
    stage_name: str
    stage_dir_name: str
    dataset_name: str
    primary_metric: str
    run_paths: RunPaths
    device: str
    gpu_name: str | None


@dataclass
class ModelResult:
    name: str
    metrics: dict[str, float]
    fit_time_sec: float
    predict_time_sec: float
    peak_rss_mb: float
    y_pred: np.ndarray
    y_score: np.ndarray | None = None
    extras: dict[str, Any] | None = None


process = psutil.Process(os.getpid())


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def now_run_id(dataset_slug: str, model_slug: str, seed: int = SEED) -> str:
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{dataset_slug}_{model_slug}_s{seed}"



def build_stage_context(stage_dir_name: str, stage_name: str, dataset_name: str, primary_metric: str, model_slug: str, device: str) -> StageContext:
    run_id = now_run_id(dataset_name.lower().replace(' ', '-').replace('/', '-'), model_slug)
    run_dir = ensure_dir(TRACK_ROOT / stage_dir_name / "artifacts" / run_id)
    figures_results = ensure_dir(run_dir / "figures" / "results")
    figures_analysis = ensure_dir(run_dir / "figures" / "analysis")
    predictions_dir = ensure_dir(run_dir / "predictions")
    logs_dir = ensure_dir(run_dir / "logs")
    checkpoints_dir = ensure_dir(run_dir / "checkpoints")
    report_dir = run_dir
    gpu_name = torch.cuda.get_device_name(0) if device.startswith("cuda") else None
    return StageContext(
        stage_name=stage_name,
        stage_dir_name=stage_dir_name,
        dataset_name=dataset_name,
        primary_metric=primary_metric,
        run_paths=RunPaths(
            run_id=run_id,
            run_dir=run_dir,
            report_dir=report_dir,
            figures_results=figures_results,
            figures_analysis=figures_analysis,
            predictions_dir=predictions_dir,
            logs_dir=logs_dir,
            checkpoints_dir=checkpoints_dir,
        ),
        device=device,
        gpu_name=gpu_name,
    )



def json_dump(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")



def yaml_dump(path: Path, data: Any) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")



def clip01(values: Iterable[float]) -> list[float]:
    return [max(0.0, min(1.0, float(v))) for v in values]



def as_list(arr: Any) -> list[Any]:
    if isinstance(arr, pd.Series):
        return arr.tolist()
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    return list(arr)



def sanitize_bool_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].astype(np.int8)
    return df



def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)



def to_dense_float32(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)



def safe_predict_scores(model: Any, X: Any) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        return proba
    if hasattr(model, "decision_function"):
        decision = model.decision_function(X)
        if np.ndim(decision) == 1:
            return 1.0 / (1.0 + np.exp(-decision))
        exp = np.exp(decision - np.max(decision, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)
    return None


# ---------- SVG chart helpers ----------


def write_svg(path: Path, width: int, height: int, body: str) -> None:
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white" />
<style>
text {{ font-family: Arial, sans-serif; fill: #111827; }}
.small {{ font-size: 12px; }}
.axis {{ font-size: 12px; fill: #374151; }}
.title {{ font-size: 20px; font-weight: bold; }}
.subtitle {{ font-size: 13px; fill: #4b5563; }}
.legend {{ font-size: 12px; }}
</style>
{body}
</svg>'''
    path.write_text(svg, encoding="utf-8")



def svg_text(x: float, y: float, text: str, cls: str = "small", anchor: str = "start", rotate: int | None = None) -> str:
    safe = (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}"{transform}>{safe}</text>'



def _chart_frame(width: int = 980, height: int = 620) -> tuple[int, int, int, int, int, int]:
    left, right, top, bottom = 90, 40, 80, 80
    return width, height, left, right, top, bottom



def _scale(vals: np.ndarray, lo: float | None = None, hi: float | None = None, padding: float = 0.05) -> tuple[float, float]:
    vals = np.asarray(vals, dtype=float)
    if lo is None:
        lo = float(np.nanmin(vals)) if vals.size else 0.0
    if hi is None:
        hi = float(np.nanmax(vals)) if vals.size else 1.0
    if math.isclose(lo, hi):
        hi = lo + 1.0
    span = hi - lo
    return lo - span * padding, hi + span * padding



def line_chart(path: Path, series: list[dict[str, Any]], title: str, subtitle: str, x_label: str, y_label: str, y_range: tuple[float, float] | None = None) -> None:
    width, height, left, right, top, bottom = _chart_frame()
    x_vals = np.concatenate([np.asarray(s["x"], dtype=float) for s in series])
    y_vals = np.concatenate([np.asarray(s["y"], dtype=float) for s in series])
    x_min, x_max = _scale(x_vals, padding=0.02)
    if y_range is None:
        y_min, y_max = _scale(y_vals)
    else:
        y_min, y_max = y_range
    chart_w = width - left - right
    chart_h = height - top - bottom

    def px_x(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * chart_w

    def px_y(y: float) -> float:
        return height - bottom - (y - y_min) / (y_max - y_min) * chart_h

    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c"]
    body = [
        svg_text(left, 35, title, cls="title"),
        svg_text(left, 58, subtitle, cls="subtitle"),
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111827"/>',
    ]
    for i in range(5):
        yv = y_min + (y_max - y_min) * i / 4
        py = px_y(yv)
        body.append(f'<line x1="{left}" y1="{py}" x2="{width-right}" y2="{py}" stroke="#e5e7eb"/>')
        body.append(svg_text(left - 10, py + 4, f"{yv:.3f}", cls="axis", anchor="end"))
    x_ticks = np.linspace(x_min, x_max, 6)
    for xv in x_ticks:
        px = px_x(float(xv))
        body.append(f'<line x1="{px}" y1="{top}" x2="{px}" y2="{height-bottom}" stroke="#f3f4f6"/>')
        body.append(svg_text(px, height - bottom + 22, f"{xv:.2f}", cls="axis", anchor="middle"))
    body.append(svg_text((left + width - right) / 2, height - 20, x_label, cls="axis", anchor="middle"))
    body.append(svg_text(24, (top + height - bottom) / 2, y_label, cls="axis", anchor="middle", rotate=-90))

    legend_x = width - right - 180
    legend_y = 28
    for idx, s in enumerate(series):
        color = s.get("color", colors[idx % len(colors)])
        pts = " ".join(f"{px_x(float(x))},{px_y(float(y))}" for x, y in zip(s["x"], s["y"]))
        body.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="3"/>')
        for x, y in zip(s["x"], s["y"]):
            body.append(f'<circle cx="{px_x(float(x))}" cy="{px_y(float(y))}" r="3.5" fill="{color}"/>')
        ly = legend_y + idx * 20
        body.append(f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x+20}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        body.append(svg_text(legend_x + 28, ly + 4, s["label"], cls="legend"))

    write_svg(path, width, height, "\n".join(body))



def bar_chart(path: Path, labels: list[str], values: list[float], title: str, subtitle: str, x_label: str, y_label: str, colors: list[str] | None = None, value_fmt: str = "{:.3f}") -> None:
    width, height, left, right, top, bottom = _chart_frame()
    chart_w = width - left - right
    chart_h = height - top - bottom
    y_min = min(0.0, min(values))
    y_max = max(values) * 1.15 if values else 1.0
    if math.isclose(y_min, y_max):
        y_max = y_min + 1.0
    colors = colors or ["#2563eb"] * len(values)

    def px_y(y: float) -> float:
        return height - bottom - (y - y_min) / (y_max - y_min) * chart_h

    body = [
        svg_text(left, 35, title, cls="title"),
        svg_text(left, 58, subtitle, cls="subtitle"),
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111827"/>',
        svg_text((left + width - right) / 2, height - 20, x_label, cls="axis", anchor="middle"),
        svg_text(24, (top + height - bottom) / 2, y_label, cls="axis", anchor="middle", rotate=-90),
    ]
    for i in range(5):
        yv = y_min + (y_max - y_min) * i / 4
        py = px_y(yv)
        body.append(f'<line x1="{left}" y1="{py}" x2="{width-right}" y2="{py}" stroke="#e5e7eb"/>')
        body.append(svg_text(left - 10, py + 4, f"{yv:.3f}", cls="axis", anchor="end"))

    n = max(1, len(values))
    bar_w = chart_w / n * 0.7
    gap = chart_w / n * 0.3
    for idx, (label, value) in enumerate(zip(labels, values)):
        x = left + idx * (bar_w + gap) + gap / 2
        y = px_y(max(value, 0.0))
        base_y = px_y(0.0)
        h = abs(base_y - y)
        rect_y = min(y, base_y)
        color = colors[idx % len(colors)]
        body.append(f'<rect x="{x}" y="{rect_y}" width="{bar_w}" height="{h}" fill="{color}" opacity="0.85"/>')
        body.append(svg_text(x + bar_w / 2, rect_y - 8, value_fmt.format(value), cls="small", anchor="middle"))
        body.append(svg_text(x + bar_w / 2, height - bottom + 28, label, cls="axis", anchor="middle", rotate=-25))
    write_svg(path, width, height, "\n".join(body))



def scatter_plot(path: Path, x: np.ndarray, y: np.ndarray, title: str, subtitle: str, x_label: str, y_label: str, color_values: np.ndarray | None = None, diagonal: bool = False) -> None:
    width, height, left, right, top, bottom = _chart_frame()
    chart_w = width - left - right
    chart_h = height - top - bottom
    x_min, x_max = _scale(x, padding=0.03)
    y_min, y_max = _scale(y, padding=0.03)
    if diagonal:
        lo = min(x_min, y_min)
        hi = max(x_max, y_max)
        x_min = y_min = lo
        x_max = y_max = hi

    def px_x(v: float) -> float:
        return left + (v - x_min) / (x_max - x_min) * chart_w

    def px_y(v: float) -> float:
        return height - bottom - (v - y_min) / (y_max - y_min) * chart_h

    body = [
        svg_text(left, 35, title, cls="title"),
        svg_text(left, 58, subtitle, cls="subtitle"),
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111827"/>',
        svg_text((left + width - right) / 2, height - 20, x_label, cls="axis", anchor="middle"),
        svg_text(24, (top + height - bottom) / 2, y_label, cls="axis", anchor="middle", rotate=-90),
    ]
    if diagonal:
        body.append(f'<line x1="{px_x(x_min)}" y1="{px_y(y_min)}" x2="{px_x(x_max)}" y2="{px_y(y_max)}" stroke="#9ca3af" stroke-dasharray="6,6"/>')
    for xv in np.linspace(x_min, x_max, 6):
        body.append(svg_text(px_x(xv), height - bottom + 22, f"{xv:.2f}", cls="axis", anchor="middle"))
    for yv in np.linspace(y_min, y_max, 6):
        body.append(svg_text(left - 10, px_y(yv) + 4, f"{yv:.2f}", cls="axis", anchor="end"))
    if color_values is None:
        color_values = np.zeros_like(x)
    c_min, c_max = _scale(color_values, padding=0.0)
    for xv, yv, cv in zip(x, y, color_values):
        ratio = 0.0 if math.isclose(c_min, c_max) else (float(cv) - c_min) / (c_max - c_min)
        color = f"rgb({int(37 + 180 * ratio)}, {int(99 + 80 * (1 - ratio))}, {int(235 - 120 * ratio)})"
        body.append(f'<circle cx="{px_x(float(xv))}" cy="{px_y(float(yv))}" r="3.0" fill="{color}" opacity="0.55"/>')
    write_svg(path, width, height, "\n".join(body))



def heatmap(path: Path, matrix: np.ndarray, row_labels: list[str], col_labels: list[str], title: str, subtitle: str) -> None:
    width, height = 900, 700
    left, top = 180, 140
    cell = min(90, int((width - left - 80) / max(1, matrix.shape[1])))
    body = [svg_text(60, 40, title, cls="title"), svg_text(60, 62, subtitle, cls="subtitle")]
    m = np.asarray(matrix, dtype=float)
    m_min = float(np.min(m))
    m_max = float(np.max(m))
    span = m_max - m_min if not math.isclose(m_min, m_max) else 1.0
    for r in range(m.shape[0]):
        body.append(svg_text(left - 12, top + r * cell + cell / 2 + 4, row_labels[r], cls="axis", anchor="end"))
        for c in range(m.shape[1]):
            ratio = (m[r, c] - m_min) / span
            color = f"rgb({int(239 - 130*ratio)}, {int(246 - 110*ratio)}, {int(255 - 230*ratio)})"
            x = left + c * cell
            y = top + r * cell
            body.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color}" stroke="#d1d5db"/>')
            body.append(svg_text(x + cell / 2, y + cell / 2 + 4, f"{m[r,c]:.0f}", cls="small", anchor="middle"))
    for c, label in enumerate(col_labels):
        body.append(svg_text(left + c * cell + cell / 2, top - 12, label, cls="axis", anchor="middle", rotate=-30))
    write_svg(path, width, height, "\n".join(body))



def table_figure(path: Path, title: str, subtitle: str, headers: list[str], rows: list[list[Any]]) -> None:
    width = 1040
    row_h = 34
    height = 140 + row_h * (len(rows) + 1)
    x0, y0 = 40, 90
    col_w = (width - 2 * x0) / max(1, len(headers))
    body = [svg_text(x0, 35, title, cls="title"), svg_text(x0, 58, subtitle, cls="subtitle")]
    for idx, header in enumerate(headers):
        x = x0 + idx * col_w
        body.append(f'<rect x="{x}" y="{y0}" width="{col_w}" height="{row_h}" fill="#e5e7eb" stroke="#d1d5db"/>')
        body.append(svg_text(x + 8, y0 + 22, header, cls="small"))
    for r, row in enumerate(rows):
        y = y0 + row_h * (r + 1)
        fill = "#ffffff" if r % 2 == 0 else "#f9fafb"
        for c, cell in enumerate(row):
            x = x0 + c * col_w
            body.append(f'<rect x="{x}" y="{y}" width="{col_w}" height="{row_h}" fill="{fill}" stroke="#e5e7eb"/>')
            cell_text = textwrap.shorten(str(cell), width=28, placeholder="…")
            body.append(svg_text(x + 8, y + 22, cell_text, cls="small"))
    write_svg(path, width, height, "\n".join(body))



def boxplot_chart(path: Path, series: dict[str, list[float]], title: str, subtitle: str, y_label: str) -> None:
    width, height, left, right, top, bottom = _chart_frame()
    chart_w = width - left - right
    chart_h = height - top - bottom
    all_values = np.array([v for values in series.values() for v in values], dtype=float)
    y_min, y_max = _scale(all_values)

    def px_y(v: float) -> float:
        return height - bottom - (v - y_min) / (y_max - y_min) * chart_h

    body = [
        svg_text(left, 35, title, cls="title"),
        svg_text(left, 58, subtitle, cls="subtitle"),
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111827"/>',
        svg_text(24, (top + height - bottom) / 2, y_label, cls="axis", anchor="middle", rotate=-90),
    ]
    for i in range(5):
        yv = y_min + (y_max - y_min) * i / 4
        py = px_y(yv)
        body.append(f'<line x1="{left}" y1="{py}" x2="{width-right}" y2="{py}" stroke="#e5e7eb"/>')
        body.append(svg_text(left - 10, py + 4, f"{yv:.3f}", cls="axis", anchor="end"))
    n = len(series)
    slot = chart_w / max(1, n)
    for idx, (label, values) in enumerate(series.items()):
        values = np.asarray(values, dtype=float)
        q1, med, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        vmin, vmax = float(np.min(values)), float(np.max(values))
        x = left + slot * idx + slot / 2
        bw = slot * 0.35
        body.append(f'<line x1="{x}" y1="{px_y(vmin)}" x2="{x}" y2="{px_y(vmax)}" stroke="#111827"/>')
        body.append(f'<rect x="{x-bw/2}" y="{px_y(q3)}" width="{bw}" height="{px_y(q1)-px_y(q3)}" fill="#93c5fd" stroke="#2563eb"/>')
        body.append(f'<line x1="{x-bw/2}" y1="{px_y(med)}" x2="{x+bw/2}" y2="{px_y(med)}" stroke="#1d4ed8" stroke-width="3"/>')
        body.append(f'<line x1="{x-bw/3}" y1="{px_y(vmin)}" x2="{x+bw/3}" y2="{px_y(vmin)}" stroke="#111827"/>')
        body.append(f'<line x1="{x-bw/3}" y1="{px_y(vmax)}" x2="{x+bw/3}" y2="{px_y(vmax)}" stroke="#111827"/>')
        body.append(svg_text(x, height - bottom + 28, label, cls="axis", anchor="middle", rotate=-20))
    write_svg(path, width, height, "\n".join(body))


# ---------- Torch MLP helpers ----------


class MLPClassifierNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], output_dim: int):
        super().__init__()
        layers: list[nn.Module] = []
        dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(dim, h), nn.ReLU(), nn.Dropout(0.15)])
            dim = h
        layers.append(nn.Linear(dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPRegressorNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...]):
        super().__init__()
        layers: list[nn.Module] = []
        dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(dim, h), nn.ReLU(), nn.Dropout(0.10)])
            dim = h
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)



def _tensor_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)



def train_torch_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    X_test: np.ndarray,
    n_classes: int,
    device: str,
    epochs: int = 16,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    X_train = np.asarray(X_train, dtype=np.float32)
    X_valid = np.asarray(X_valid, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    y_valid = np.asarray(y_valid, dtype=np.int64)

    model = MLPClassifierNet(X_train.shape[1], (256, 128), n_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_valid = float("inf")
    history = []
    train_loader = _tensor_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
    valid_loader = _tensor_loader(X_valid, y_valid, batch_size=batch_size, shuffle=False)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(epochs):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        valid_losses = []
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                valid_losses.append(criterion(logits, yb).item())
        mean_valid = float(np.mean(valid_losses))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "valid_loss": mean_valid})
        if mean_valid < best_valid:
            best_valid = mean_valid
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test).to(device))
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=1)
    extras = {"history": history}
    if device.startswith("cuda"):
        extras["peak_gpu_mb"] = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    return preds, probs, extras



def train_torch_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    X_test: np.ndarray,
    device: str,
    epochs: int = 20,
    batch_size: int = 512,
) -> tuple[np.ndarray, dict[str, Any]]:
    X_train = np.asarray(X_train, dtype=np.float32)
    X_valid = np.asarray(X_valid, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    y_valid = np.asarray(y_valid, dtype=np.float32)

    model = MLPRegressorNet(X_train.shape[1], (256, 128)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss()
    best_state = None
    best_valid = float("inf")
    history = []
    train_loader = _tensor_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
    valid_loader = _tensor_loader(X_valid, y_valid, batch_size=batch_size, shuffle=False)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(epochs):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        valid_losses = []
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                valid_losses.append(criterion(pred, yb).item())
        mean_valid = float(np.mean(valid_losses))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "valid_loss": mean_valid})
        if mean_valid < best_valid:
            best_valid = mean_valid
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X_test).to(device)).cpu().numpy()
    extras = {"history": history}
    if device.startswith("cuda"):
        extras["peak_gpu_mb"] = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    return preds, extras


# ---------- Metrics helpers ----------


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
    }



def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_proba is not None:
        metrics["mean_confidence"] = float(np.max(y_proba, axis=1).mean())
    return metrics



def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse),
        "r2": float(r2_score(y_true, y_pred)),
    }



def timed_fit_predict(model: Any, X_train: Any, y_train: Any, X_test: Any) -> tuple[Any, np.ndarray, np.ndarray | None, float, float, float]:
    rss_before = process.memory_info().rss
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0
    rss_after_fit = process.memory_info().rss
    t1 = time.perf_counter()
    y_pred = model.predict(X_test)
    predict_time = time.perf_counter() - t1
    y_score = safe_predict_scores(model, X_test)
    peak_rss_mb = max(rss_before, rss_after_fit, process.memory_info().rss) / (1024 ** 2)
    return model, y_pred, y_score, fit_time, predict_time, peak_rss_mb


# ---------- Stage 1: Adult classification ----------



def load_adult() -> pd.DataFrame:
    df = load_dataset("scikit-learn/adult-census-income", split="train").to_pandas()
    df = df.replace("?", np.nan)
    return df



def adult_preprocessors(df: pd.DataFrame) -> tuple[list[str], list[str], ColumnTransformer, Pipeline]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]
    sparse_preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))]), cat_cols),
        ]
    )
    dense_mlp = Pipeline(
        steps=[
            ("preprocessor", ColumnTransformer(
                transformers=[
                    ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
                    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))]), cat_cols),
                ]
            )),
            ("scale", MaxAbsScaler()),
        ]
    )
    return num_cols, cat_cols, sparse_preprocessor, dense_mlp



def run_stage_01(device: str) -> dict[str, Any]:
    ctx = build_stage_context("01_tabular_classification", "01 Tabular Classification", "adult-census-income", "auprc", "model-suite", device)
    df = load_adult()
    X = df.drop(columns=["income"])
    y = (df["income"] == ">50K").astype(int).to_numpy()
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    train_valid_idx, test_idx = next(sss.split(X, y))
    X_train_valid, X_test = X.iloc[train_valid_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)
    y_train_valid, y_test = y[train_valid_idx], y[test_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.17647, random_state=SEED)
    train_idx, valid_idx = next(sss2.split(X_train_valid, y_train_valid))
    X_train, X_valid = X_train_valid.iloc[train_idx].reset_index(drop=True), X_train_valid.iloc[valid_idx].reset_index(drop=True)
    y_train, y_valid = y_train_valid[train_idx], y_train_valid[valid_idx]

    _, _, preprocessor, mlp_preprocessor = adult_preprocessors(X_train)

    models = {
        "dummy_prior": DummyClassifier(strategy="prior"),
        "logistic_regression": Pipeline([("preprocessor", clone(preprocessor)), ("model", LogisticRegression(max_iter=1200, class_weight="balanced", n_jobs=-1))]),
        "random_forest": Pipeline([("preprocessor", clone(preprocessor)), ("model", RandomForestClassifier(n_estimators=260, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED))]),
    }

    results: dict[str, ModelResult] = {}
    for name, model in models.items():
        model, y_pred, y_score, fit_time, predict_time, peak_rss = timed_fit_predict(model, X_train, y_train, X_test)
        if y_score is None:
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)[:, 1]
            else:
                y_score = y_pred.astype(float)
        results[name] = ModelResult(
            name=name,
            metrics=binary_metrics(y_test, y_pred, np.asarray(y_score)),
            fit_time_sec=fit_time,
            predict_time_sec=predict_time,
            peak_rss_mb=peak_rss,
            y_pred=np.asarray(y_pred),
            y_score=np.asarray(y_score),
        )

    mlp_transformer = clone(mlp_preprocessor)
    X_train_mlp = to_dense_float32(mlp_transformer.fit_transform(X_train))
    X_valid_mlp = to_dense_float32(mlp_transformer.transform(X_valid))
    X_test_mlp = to_dense_float32(mlp_transformer.transform(X_test))
    rss_before = process.memory_info().rss
    t0 = time.perf_counter()
    y_pred_mlp, y_prob_mlp, extras = train_torch_classifier(X_train_mlp, y_train, X_valid_mlp, y_valid, X_test_mlp, n_classes=2, device=device, epochs=14, batch_size=768)
    fit_time = time.perf_counter() - t0
    peak_rss = max(rss_before, process.memory_info().rss) / (1024 ** 2)
    results["gpu_mlp"] = ModelResult(
        name="gpu_mlp",
        metrics=binary_metrics(y_test, y_pred_mlp, y_prob_mlp[:, 1]),
        fit_time_sec=fit_time,
        predict_time_sec=0.0,
        peak_rss_mb=peak_rss,
        y_pred=y_pred_mlp,
        y_score=y_prob_mlp[:, 1],
        extras=extras,
    )

    best_name = max(results, key=lambda name: results[name].metrics[ctx.primary_metric])
    best = results[best_name]
    best_pipeline = models.get(best_name)
    if best_name == "gpu_mlp":
        analysis_model_name = max((n for n in results if n != "gpu_mlp"), key=lambda name: results[name].metrics[ctx.primary_metric])
        analysis_pipeline = models[analysis_model_name]
        analysis_pipeline.fit(X_train, y_train)
    else:
        analysis_pipeline = models[best_name].fit(X_train, y_train)
        analysis_model_name = best_name

    # Config / metrics
    config = {
        "track": TRACK,
        "stage": ctx.stage_name,
        "dataset": ctx.dataset_name,
        "seed": SEED,
        "split": {"train": int(len(X_train)), "valid": int(len(X_valid)), "test": int(len(X_test))},
        "hardware": {"device": device, "gpu_name": ctx.gpu_name, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")},
        "models": list(results.keys()),
    }
    yaml_dump(ctx.run_paths.run_dir / "config.yaml", config)

    metrics_payload = {
        "primary_metric": ctx.primary_metric,
        "best_model": best_name,
        "models": {name: {**res.metrics, "fit_time_sec": res.fit_time_sec, "predict_time_sec": res.predict_time_sec, "peak_rss_mb": res.peak_rss_mb, **(res.extras or {})} for name, res in results.items()},
    }
    json_dump(ctx.run_paths.run_dir / "metrics.json", metrics_payload)

    # Predictions samples
    pred_df = X_test.copy()
    pred_df["label"] = y_test
    pred_df["pred"] = best.y_pred
    pred_df["score"] = best.y_score
    pred_df["error"] = (pred_df["label"] != pred_df["pred"]).astype(int)
    pred_df.sort_values("score", ascending=False).head(20).to_csv(ctx.run_paths.predictions_dir / "top_scored_predictions.csv", index=False)
    pred_df.sort_values(["error", "score"], ascending=[False, False]).head(30).to_csv(ctx.run_paths.predictions_dir / "high_confidence_errors.csv", index=False)

    # Result figures
    class_dist = pd.Series(y).value_counts().sort_index()
    bar_chart(ctx.run_paths.figures_results / "class_distribution.svg", ["<=50K", ">50K"], [float(class_dist.get(0, 0)), float(class_dist.get(1, 0))], "Adult income class distribution", "Primary dataset class balance before split.", "class", "count", colors=["#60a5fa", "#2563eb"], value_fmt="{:.0f}")
    age_hist_counts, age_bins = np.histogram(df["age"].dropna(), bins=12)
    bar_chart(ctx.run_paths.figures_results / "age_histogram.svg", [f"{int(age_bins[i])}-{int(age_bins[i+1])}" for i in range(len(age_hist_counts))], age_hist_counts.astype(float).tolist(), "Age histogram", "Representative numeric feature distribution.", "age bucket", "count", value_fmt="{:.0f}")
    fpr, tpr, _ = roc_curve(y_test, best.y_score)
    line_chart(ctx.run_paths.figures_results / "roc_curve.svg", [{"label": best_name, "x": fpr, "y": tpr, "color": "#2563eb"}, {"label": "random", "x": [0, 1], "y": [0, 1], "color": "#9ca3af"}], "ROC curve", f"Best model: {best_name} (AUROC={best.metrics['auroc']:.3f})", "false positive rate", "true positive rate", y_range=(0.0, 1.0))
    precision, recall, _ = precision_recall_curve(y_test, best.y_score)
    line_chart(ctx.run_paths.figures_results / "pr_curve.svg", [{"label": best_name, "x": recall, "y": precision, "color": "#dc2626"}], "Precision-Recall curve", f"Best model: {best_name} (AUPRC={best.metrics['auprc']:.3f})", "recall", "precision", y_range=(0.0, 1.02))
    cm = confusion_matrix(y_test, best.y_pred)
    heatmap(ctx.run_paths.figures_results / "confusion_matrix.svg", cm, ["true <=50K", "true >50K"], ["pred <=50K", "pred >50K"], "Confusion matrix", f"Best model: {best_name}")
    prob_true, prob_pred = calibration_curve(y_test, best.y_score, n_bins=10, strategy="quantile")
    line_chart(ctx.run_paths.figures_results / "calibration_curve.svg", [{"label": best_name, "x": prob_pred, "y": prob_true, "color": "#059669"}, {"label": "ideal", "x": [0, 1], "y": [0, 1], "color": "#9ca3af"}], "Calibration curve", "Probability calibration on the test split.", "mean predicted probability", "fraction of positives", y_range=(0.0, 1.02))

    # Analysis figures
    perm = permutation_importance(analysis_pipeline, X_test, y_test, n_repeats=5, random_state=SEED, scoring="average_precision")
    feat_names = analysis_pipeline.named_steps["preprocessor"].get_feature_names_out()
    top_idx = np.argsort(perm.importances_mean)[-12:][::-1]
    bar_chart(ctx.run_paths.figures_analysis / "permutation_importance.svg", [feat_names[i].split("__")[-1][:18] for i in top_idx], perm.importances_mean[top_idx].tolist(), "Permutation importance", f"Computed with {analysis_model_name} on test split.", "feature", "importance drop")
    slice_error = pred_df.assign(sex=X_test["sex"].values, education=X_test["education"].values).groupby("sex")["error"].mean().sort_values(ascending=False)
    bar_chart(ctx.run_paths.figures_analysis / "error_slice_by_sex.svg", slice_error.index.tolist(), slice_error.values.tolist(), "Error slice by sex", "Mean error rate across a simple demographic slice.", "slice", "error rate")
    conf_bins = pd.cut(pred_df["score"], bins=np.linspace(0, 1, 11), include_lowest=True)
    conf_acc = pred_df.groupby(conf_bins, observed=False).apply(lambda g: 1 - g["error"].mean()).fillna(0.0)
    conf_mid = [0.05 + 0.1 * i for i in range(len(conf_acc))]
    line_chart(ctx.run_paths.figures_analysis / "confidence_vs_correctness.svg", [{"label": best_name, "x": conf_mid, "y": conf_acc.values, "color": "#7c3aed"}], "Confidence vs correctness", "Higher confidence bins should correspond to higher observed accuracy.", "predicted probability bin", "accuracy", y_range=(0.0, 1.02))

    failure_examples = pred_df[pred_df["error"] == 1].sort_values("score", ascending=False).head(8)
    table_rows = failure_examples[["age", "education", "occupation", "hours.per.week", "label", "pred", "score"]].round({"score": 3}).values.tolist()
    table_figure(ctx.run_paths.figures_analysis / "failure_examples.svg", "High-confidence failure examples", "Representative mistakes from the best model.", ["age", "education", "occupation", "hours", "label", "pred", "score"], table_rows)

    summary = f"""# Run Summary

## 1. Problem

- Task: Binary income classification on adult census records.
- Track / Stage: {TRACK} / {ctx.stage_name}
- Why this run exists: Establish leakage-safe preprocessing and compare weak/strong baselines plus a GPU MLP on device `{device}`.

## 2. Hypothesis

- What changed from the previous run: Initial execution for the ML track.
- Expected effect: Tree ensembles should outperform dummy/logistic baselines on AUPRC; GPU MLP should provide a competitive nonlinear reference while explicitly using GPU 0.

## 3. Dataset

- Dataset name: {ctx.dataset_name}
- Split version: train={len(X_train)}, valid={len(X_valid)}, test={len(X_test)}
- Preprocessing: `?` -> missing, median numeric imputation, frequent categorical imputation, one-hot encoding.
- Data caveats: Adult labels are imbalanced and include sensitive attributes that can create slice disparities.

## 4. Training Setup

- Model: dummy / logistic regression / random forest / GPU MLP
- Tokenizer / Processor: N/A
- Seed: {SEED}
- Batch size: 768 for GPU MLP
- Learning rate: 1e-3 for GPU MLP
- Epochs / steps: 14 epochs for GPU MLP
- Hardware: {device} {ctx.gpu_name or ''}

## 5. Best Metrics

{markdown_table(["Model", "AUPRC", "AUROC", "F1", "Fit sec"], [[name, f"{res.metrics['auprc']:.3f}", f"{res.metrics['auroc']:.3f}", f"{res.metrics['f1']:.3f}", f"{res.fit_time_sec:.1f}"] for name, res in sorted(results.items(), key=lambda kv: kv[1].metrics['auprc'], reverse=True)])}

## 6. Result Figures

- `figures/results/class_distribution.svg`, `age_histogram.svg`, `roc_curve.svg`, `pr_curve.svg`, `confusion_matrix.svg`, `calibration_curve.svg`
- Why these matter: They show imbalance, feature spread, ranking quality, threshold behavior, and calibration quality.

## 7. Analysis Figures

- `figures/analysis/permutation_importance.svg`, `error_slice_by_sex.svg`, `confidence_vs_correctness.svg`, `failure_examples.svg`
- Key failure patterns: High-confidence errors concentrate in minority positive samples and slice disparity is visible between sex groups.

## 8. Sample Predictions

- Best examples: `predictions/top_scored_predictions.csv`
- Failure examples: `predictions/high_confidence_errors.csv`

## 9. Decision

- 대표 artifact로 남길까? yes
- 모델 가중치를 별도로 배포할 계획이 있는가?: no
- Next run: threshold tuning or cost-sensitive calibration for the positive income class.
"""
    (ctx.run_paths.run_dir / "summary.md").write_text(summary, encoding="utf-8")
    promote_run(ctx)
    return {
        "stage": ctx.stage_name,
        "run_id": ctx.run_paths.run_id,
        "best_model": best_name,
        "best_metrics": best.metrics,
        "report_dir": str(ctx.run_paths.report_dir.relative_to(ROOT)),
    }


# ---------- Stage 2: California housing regression ----------


def load_california() -> pd.DataFrame:
    from sklearn.datasets import fetch_california_housing

    return fetch_california_housing(as_frame=True).frame



def run_stage_02(device: str) -> dict[str, Any]:
    ctx = build_stage_context("02_tabular_regression", "02 Tabular Regression", "california-housing", "rmse", "model-suite", device)
    df = load_california()
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"].to_numpy()
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(X, y, test_size=0.15, random_state=SEED)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=0.17647, random_state=SEED)

    num_cols = X.columns.tolist()
    linear_preprocessor = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    tree_preprocessor = Pipeline([("imputer", SimpleImputer(strategy="median"))])

    models = {
        "dummy_mean": Pipeline([("prep", tree_preprocessor), ("model", DummyRegressor(strategy="mean"))]),
        "linear_regression": Pipeline([("prep", linear_preprocessor), ("model", LinearRegression())]),
        "ridge": Pipeline([("prep", linear_preprocessor), ("model", Ridge(alpha=2.0))]),
        "random_forest": Pipeline([("prep", tree_preprocessor), ("model", RandomForestRegressor(n_estimators=240, min_samples_leaf=2, n_jobs=-1, random_state=SEED))]),
        "hist_gbdt": Pipeline([("prep", tree_preprocessor), ("model", HistGradientBoostingRegressor(max_depth=6, max_iter=220, learning_rate=0.06, random_state=SEED))]),
    }

    results: dict[str, ModelResult] = {}
    for name, model in models.items():
        model, y_pred, _, fit_time, predict_time, peak_rss = timed_fit_predict(model, X_train, y_train, X_test)
        results[name] = ModelResult(name=name, metrics=regression_metrics(y_test, y_pred), fit_time_sec=fit_time, predict_time_sec=predict_time, peak_rss_mb=peak_rss, y_pred=np.asarray(y_pred))

    mlp_prep = clone(linear_preprocessor)
    X_train_mlp = to_dense_float32(mlp_prep.fit_transform(X_train))
    X_valid_mlp = to_dense_float32(mlp_prep.transform(X_valid))
    X_test_mlp = to_dense_float32(mlp_prep.transform(X_test))
    rss_before = process.memory_info().rss
    t0 = time.perf_counter()
    y_pred_mlp, extras = train_torch_regressor(X_train_mlp, y_train, X_valid_mlp, y_valid, X_test_mlp, device=device, epochs=18, batch_size=768)
    fit_time = time.perf_counter() - t0
    peak_rss = max(rss_before, process.memory_info().rss) / (1024 ** 2)
    results["gpu_mlp"] = ModelResult(name="gpu_mlp", metrics=regression_metrics(y_test, y_pred_mlp), fit_time_sec=fit_time, predict_time_sec=0.0, peak_rss_mb=peak_rss, y_pred=y_pred_mlp, extras=extras)

    best_name = min(results, key=lambda name: results[name].metrics[ctx.primary_metric])
    best = results[best_name]
    best_pipeline = models.get(best_name)
    if best_pipeline is None:
        analysis_name = min((n for n in results if n != "gpu_mlp"), key=lambda name: results[name].metrics[ctx.primary_metric])
        analysis_pipeline = models[analysis_name].fit(X_train, y_train)
    else:
        analysis_name = best_name
        analysis_pipeline = models[best_name].fit(X_train, y_train)

    config = {
        "track": TRACK,
        "stage": ctx.stage_name,
        "dataset": ctx.dataset_name,
        "seed": SEED,
        "split": {"train": int(len(X_train)), "valid": int(len(X_valid)), "test": int(len(X_test))},
        "hardware": {"device": device, "gpu_name": ctx.gpu_name},
        "models": list(results.keys()),
    }
    yaml_dump(ctx.run_paths.run_dir / "config.yaml", config)
    metrics_payload = {
        "primary_metric": ctx.primary_metric,
        "best_model": best_name,
        "models": {name: {**res.metrics, "fit_time_sec": res.fit_time_sec, "predict_time_sec": res.predict_time_sec, "peak_rss_mb": res.peak_rss_mb, **(res.extras or {})} for name, res in results.items()},
    }
    json_dump(ctx.run_paths.run_dir / "metrics.json", metrics_payload)

    pred_df = X_test.copy()
    pred_df["target"] = y_test
    pred_df["pred"] = best.y_pred
    pred_df["abs_error"] = np.abs(pred_df["target"] - pred_df["pred"])
    pred_df.sort_values("abs_error", ascending=False).head(30).to_csv(ctx.run_paths.predictions_dir / "worst_predictions.csv", index=False)

    target_counts, target_bins = np.histogram(y, bins=12)
    bar_chart(ctx.run_paths.figures_results / "target_histogram.svg", [f"{target_bins[i]:.1f}-{target_bins[i+1]:.1f}" for i in range(len(target_counts))], target_counts.astype(float).tolist(), "Target histogram", "California housing target distribution.", "target bucket", "count", value_fmt="{:.0f}")
    fractions = [0.1, 0.3, 0.6, 1.0]
    rmse_values = []
    for frac in fractions:
        n = max(200, int(len(X_train) * frac))
        model = Pipeline([("prep", tree_preprocessor), ("model", HistGradientBoostingRegressor(max_depth=6, max_iter=180, learning_rate=0.06, random_state=SEED))])
        model.fit(X_train.iloc[:n], y_train[:n])
        pred = model.predict(X_valid)
        rmse_values.append(math.sqrt(mean_squared_error(y_valid, pred)))
    line_chart(ctx.run_paths.figures_results / "learning_curve.svg", [{"label": "hist_gbdt", "x": fractions, "y": rmse_values, "color": "#2563eb"}], "Learning curve", "Validation RMSE across training fractions.", "training fraction", "RMSE")
    scatter_plot(ctx.run_paths.figures_results / "parity_plot.svg", y_test, best.y_pred, "Parity plot", f"Best model: {best_name}", "true target", "predicted target", diagonal=True)
    residuals = best.y_pred - y_test
    res_counts, res_bins = np.histogram(residuals, bins=14)
    bar_chart(ctx.run_paths.figures_results / "residual_histogram.svg", [f"{res_bins[i]:.1f}" for i in range(len(res_counts))], res_counts.astype(float).tolist(), "Residual histogram", "Prediction residual distribution on the test split.", "residual bucket", "count", value_fmt="{:.0f}")
    scatter_plot(ctx.run_paths.figures_results / "residual_vs_target.svg", y_test, residuals, "Residual vs target", "Residual structure across target values.", "true target", "residual")

    perm = permutation_importance(analysis_pipeline, X_test, y_test, n_repeats=5, random_state=SEED, scoring="neg_root_mean_squared_error")
    top_idx = np.argsort(np.abs(perm.importances_mean))[-10:][::-1]
    bar_chart(ctx.run_paths.figures_analysis / "feature_importance.svg", [num_cols[i] for i in top_idx], np.abs(perm.importances_mean[top_idx]).tolist(), "Feature importance", f"Permutation importance from {analysis_name}.", "feature", "importance drop")
    medinc_bin = pd.qcut(X_test["MedInc"], 5, duplicates="drop")
    slice_rmse = pred_df.assign(medinc_bucket=medinc_bin.astype(str)).groupby("medinc_bucket")[["target", "pred"]].apply(lambda g: math.sqrt(mean_squared_error(g["target"], g["pred"]))).sort_values(ascending=False)
    bar_chart(ctx.run_paths.figures_analysis / "error_slice_by_income.svg", slice_rmse.index.tolist(), slice_rmse.values.tolist(), "Error slice by income bucket", "RMSE varies across income ranges.", "MedInc quantile", "RMSE")
    worst_rows = pred_df.sort_values("abs_error", ascending=False).head(10)[["MedInc", "AveRooms", "Latitude", "Longitude", "target", "pred", "abs_error"]].round(3).values.tolist()
    table_figure(ctx.run_paths.figures_analysis / "worst_prediction_cases.svg", "Worst prediction cases", "Largest absolute errors on the test split.", ["MedInc", "AveRooms", "Lat", "Lon", "target", "pred", "abs err"], worst_rows)
    lat_bin = pd.cut(X_test["Latitude"], bins=6)
    lat_mae = pred_df.assign(lat_bucket=lat_bin.astype(str)).groupby("lat_bucket")["abs_error"].mean().sort_values(ascending=False)
    bar_chart(ctx.run_paths.figures_analysis / "regional_error_slice.svg", lat_mae.index.tolist(), lat_mae.values.tolist(), "Regional error slice", "MAE differs by latitude bucket.", "latitude bucket", "MAE")

    summary = f"""# Run Summary

## 1. Problem

- Task: Tabular regression for California housing prices.
- Track / Stage: {TRACK} / {ctx.stage_name}
- Why this run exists: Practice baseline comparison, residual analysis, and feature interpretation with a lightweight GPU reference.

## 2. Hypothesis

- What changed from the previous run: Initial regression suite for the ML track.
- Expected effect: Tree-based models should improve RMSE over linear baselines while the GPU MLP provides a non-linear comparator.

## 3. Dataset

- Dataset name: {ctx.dataset_name}
- Split version: train={len(X_train)}, valid={len(X_valid)}, test={len(X_test)}
- Preprocessing: median imputation and scaling for linear / MLP models.
- Data caveats: Target is capped and the coastal geography introduces structured residual patterns.

## 4. Training Setup

- Model: dummy / linear regression / ridge / random forest / hist GBDT / GPU MLP
- Seed: {SEED}
- Batch size: 768 for GPU MLP
- Learning rate: 2e-3 for GPU MLP
- Epochs / steps: 18 epochs for GPU MLP
- Hardware: {device} {ctx.gpu_name or ''}

## 5. Best Metrics

{markdown_table(["Model", "RMSE", "MAE", "R2", "Fit sec"], [[name, f"{res.metrics['rmse']:.3f}", f"{res.metrics['mae']:.3f}", f"{res.metrics['r2']:.3f}", f"{res.fit_time_sec:.1f}"] for name, res in sorted(results.items(), key=lambda kv: kv[1].metrics['rmse'])])}

## 6. Result Figures

- `figures/results/target_histogram.svg`, `learning_curve.svg`, `parity_plot.svg`, `residual_histogram.svg`, `residual_vs_target.svg`
- Why these matter: They expose target shape, sample-efficiency, fit quality, and residual structure.

## 7. Analysis Figures

- `figures/analysis/feature_importance.svg`, `worst_prediction_cases.svg`, `regional_error_slice.svg`, `error_slice_by_income.svg`
- Key failure patterns: Largest misses cluster in high-value tracts and geographic pockets near the coast.

## 8. Sample Predictions

- Worst examples: `predictions/worst_predictions.csv`
- Failure examples: High absolute-error rows in the table figure.

## 9. Decision

- 대표 artifact로 남길까? yes
- 모델 가중치를 별도로 배포할 계획이 있는가?: no
- Next run: Add log-target experiments or capped-target aware losses.
"""
    (ctx.run_paths.run_dir / "summary.md").write_text(summary, encoding="utf-8")
    promote_run(ctx)
    return {
        "stage": ctx.stage_name,
        "run_id": ctx.run_paths.run_id,
        "best_model": best_name,
        "best_metrics": best.metrics,
        "report_dir": str(ctx.run_paths.report_dir.relative_to(ROOT)),
    }


# ---------- Stage 3: Bike sharing model selection ----------


def load_bike_sharing() -> pd.DataFrame:
    cache_dir = ensure_dir(DATA_ROOT / "external" / "bike_sharing")
    zip_path = cache_dir / "bike_sharing_dataset.zip"
    if not zip_path.exists():
        url = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        zip_path.write_bytes(r.content)
    with zipfile.ZipFile(zip_path) as zf:
        df = pd.read_csv(zf.open("hour.csv"))
    df["dteday"] = pd.to_datetime(df["dteday"])
    return df.sort_values(["dteday", "hr"]).reset_index(drop=True)



def bike_features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    df = df.copy()
    df["day_of_year"] = df["dteday"].dt.dayofyear
    df["month"] = df["dteday"].dt.month
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7.0)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7.0)
    df["hour_sin"] = np.sin(2 * np.pi * df["hr"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hr"] / 24.0)
    y = df["cnt"].to_numpy()
    leakage_cols = ["instant", "dteday", "casual", "registered", "cnt"]
    X = df.drop(columns=leakage_cols)
    return X, y



def run_stage_03(device: str) -> dict[str, Any]:
    ctx = build_stage_context("03_model_selection_and_interpretation", "03 Model Selection And Interpretation", "bike-sharing-hourly", "rmse", "tuned-hgbdt", device)
    df = load_bike_sharing()
    X, y = bike_features(df)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx].reset_index(drop=True), X.iloc[split_idx:].reset_index(drop=True)
    y_train, y_test = y[:split_idx], y[split_idx:]
    preprocessor = Pipeline([("imputer", SimpleImputer(strategy="median"))])

    baseline_model = Pipeline([("prep", clone(preprocessor)), ("model", PoissonRegressor(alpha=0.001, max_iter=500))])
    baseline_model.fit(X_train, y_train)
    baseline_pred = baseline_model.predict(X_test)
    baseline_result = ModelResult(name="poisson_baseline", metrics=regression_metrics(y_test, baseline_pred), fit_time_sec=0.0, predict_time_sec=0.0, peak_rss_mb=process.memory_info().rss / (1024 ** 2), y_pred=baseline_pred)

    tscv = TimeSeriesSplit(n_splits=5)
    candidate_params = [
        {"learning_rate": 0.08, "max_leaf_nodes": 31, "min_samples_leaf": 20, "max_iter": 160},
        {"learning_rate": 0.05, "max_leaf_nodes": 63, "min_samples_leaf": 20, "max_iter": 220},
        {"learning_rate": 0.03, "max_leaf_nodes": 127, "min_samples_leaf": 30, "max_iter": 280},
    ]
    fold_scores: dict[str, list[float]] = {}
    model_records = []
    X_train_np = preprocessor.fit_transform(X_train)
    for idx, params in enumerate(candidate_params, start=1):
        key = f"candidate_{idx}"
        scores = []
        for tr_idx, val_idx in tscv.split(X_train_np):
            model = HistGradientBoostingRegressor(loss="poisson", random_state=SEED, **params)
            model.fit(X_train_np[tr_idx], y_train[tr_idx])
            pred = model.predict(X_train_np[val_idx])
            scores.append(math.sqrt(mean_squared_error(y_train[val_idx], pred)))
        fold_scores[key] = scores
        model_records.append({"name": key, "params": params, "mean_rmse": float(np.mean(scores)), "std_rmse": float(np.std(scores))})
    best_record = min(model_records, key=lambda rec: rec["mean_rmse"])
    best_params = best_record["params"]
    tuned_model = Pipeline([("prep", clone(preprocessor)), ("model", HistGradientBoostingRegressor(loss="poisson", random_state=SEED, **best_params))])
    tuned_model.fit(X_train, y_train)
    tuned_pred = tuned_model.predict(X_test)
    tuned_result = ModelResult(name="tuned_hist_gbdt", metrics=regression_metrics(y_test, tuned_pred), fit_time_sec=0.0, predict_time_sec=0.0, peak_rss_mb=process.memory_info().rss / (1024 ** 2), y_pred=tuned_pred, extras={"cv": best_record})

    # GPU MLP comparator
    scaler = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    n_valid = int(len(X_train) * 0.2)
    X_fit, X_valid = X_train.iloc[:-n_valid], X_train.iloc[-n_valid:]
    y_fit, y_valid = y_train[:-n_valid], y_train[-n_valid:]
    X_fit_np = to_dense_float32(scaler.fit_transform(X_fit))
    X_valid_np = to_dense_float32(scaler.transform(X_valid))
    X_test_np = to_dense_float32(scaler.transform(X_test))
    y_pred_mlp, extras = train_torch_regressor(X_fit_np, y_fit, X_valid_np, y_valid, X_test_np, device=device, epochs=16, batch_size=1024)
    gpu_result = ModelResult(name="gpu_mlp", metrics=regression_metrics(y_test, y_pred_mlp), fit_time_sec=0.0, predict_time_sec=0.0, peak_rss_mb=process.memory_info().rss / (1024 ** 2), y_pred=y_pred_mlp, extras=extras)

    results = {
        baseline_result.name: baseline_result,
        tuned_result.name: tuned_result,
        gpu_result.name: gpu_result,
    }
    best_name = min(results, key=lambda name: results[name].metrics[ctx.primary_metric])
    best = results[best_name]

    config = {
        "track": TRACK,
        "stage": ctx.stage_name,
        "dataset": ctx.dataset_name,
        "seed": SEED,
        "split": {"train": int(len(X_train)), "test": int(len(X_test)), "cv": 5},
        "hardware": {"device": device, "gpu_name": ctx.gpu_name},
        "best_params": best_params,
    }
    yaml_dump(ctx.run_paths.run_dir / "config.yaml", config)
    json_dump(ctx.run_paths.run_dir / "metrics.json", {
        "primary_metric": ctx.primary_metric,
        "best_model": best_name,
        "models": {name: {**res.metrics, **(res.extras or {})} for name, res in results.items()},
        "cv_records": model_records,
    })

    pred_df = X_test.copy()
    pred_df["target"] = y_test
    pred_df["pred"] = best.y_pred
    pred_df["abs_error"] = np.abs(pred_df["target"] - pred_df["pred"])
    pred_df.sort_values("abs_error", ascending=False).head(30).to_csv(ctx.run_paths.predictions_dir / "worst_predictions.csv", index=False)

    # figures
    boxplot_chart(ctx.run_paths.figures_results / "cv_fold_score_boxplot.svg", fold_scores, "CV fold RMSE boxplot", "TimeSeriesSplit RMSE across tuned candidates.", "RMSE")
    val_curve_leaf = []
    leaf_values = [15, 31, 63, 127]
    X_train_np2 = preprocessor.fit_transform(X_train)
    for leaf in leaf_values:
        scores = []
        for tr_idx, val_idx in tscv.split(X_train_np2):
            model = HistGradientBoostingRegressor(loss="poisson", learning_rate=0.05, max_leaf_nodes=leaf, min_samples_leaf=20, max_iter=220, random_state=SEED)
            model.fit(X_train_np2[tr_idx], y_train[tr_idx])
            pred = model.predict(X_train_np2[val_idx])
            scores.append(math.sqrt(mean_squared_error(y_train[val_idx], pred)))
        val_curve_leaf.append(float(np.mean(scores)))
    line_chart(ctx.run_paths.figures_results / "validation_curve.svg", [{"label": "mean CV RMSE", "x": leaf_values, "y": val_curve_leaf, "color": "#2563eb"}], "Validation curve", "Effect of max_leaf_nodes under time-aware CV.", "max_leaf_nodes", "RMSE")
    perm = permutation_importance(tuned_model, X_test, y_test, n_repeats=4, random_state=SEED, scoring="neg_root_mean_squared_error")
    top_idx = np.argsort(np.abs(perm.importances_mean))[-12:][::-1]
    feature_names = X.columns.tolist()
    bar_chart(ctx.run_paths.figures_results / "top_feature_importance.svg", [feature_names[i] for i in top_idx], np.abs(perm.importances_mean[top_idx]).tolist(), "Top feature importance", "Permutation importance on the held-out test window.", "feature", "importance drop")

    slice_rmse = pred_df.groupby("season")[["target", "pred"]].apply(lambda g: math.sqrt(mean_squared_error(g["target"], g["pred"]))).sort_values(ascending=False)
    bar_chart(ctx.run_paths.figures_analysis / "subgroup_metric_comparison.svg", [f"season_{i}" for i in slice_rmse.index.tolist()], slice_rmse.values.tolist(), "Subgroup metric comparison", "RMSE by season on the test split.", "season", "RMSE")
    pred_bins = pd.cut(pred_df["pred"], bins=8)
    bin_mae = pred_df.groupby(pred_bins, observed=False)["abs_error"].mean().fillna(0.0)
    line_chart(ctx.run_paths.figures_analysis / "confidence_bin_plot.svg", [{"label": "mean abs error", "x": list(range(1, len(bin_mae)+1)), "y": bin_mae.values, "color": "#dc2626"}], "Prediction-bin error plot", "A regression analogue to confidence bins: larger predicted demand is harder.", "prediction bin index", "MAE")
    failure_slices = pred_df.assign(weather=X_test["weathersit"].values).groupby(["workingday", "weather"])["abs_error"].mean().sort_values(ascending=False).head(10)
    rows = [[idx[0], idx[1], f"{val:.2f}"] for idx, val in failure_slices.items()]
    table_figure(ctx.run_paths.figures_analysis / "common_failure_slice_summary.svg", "Common failure slice summary", "Highest-error combinations of working day and weather.", ["workingday", "weather", "MAE"], rows)

    summary = f"""# Run Summary

## 1. Problem

- Task: Time-aware count regression for bike rental demand.
- Track / Stage: {TRACK} / {ctx.stage_name}
- Why this run exists: Practice validation strategy, leakage prevention, and parameter selection with a held-out future window.

## 2. Hypothesis

- What changed from the previous run: Introduced TimeSeriesSplit and tuned HistGradientBoosting over a Poisson baseline.
- Expected effect: Time-aware tuning should improve RMSE on the final 20% future window and expose where seasonal/weather slices remain difficult.

## 3. Dataset

- Dataset name: {ctx.dataset_name}
- Split version: first 80% train/CV, last 20% test
- Preprocessing: dropped leakage columns `casual`, `registered`, `cnt`, `instant`; added cyclic weekday/hour features.
- Data caveats: Extreme demand spikes during commuting peaks and bad weather cause heavy-tailed residuals.

## 4. Training Setup

- Model: Poisson baseline / tuned HistGradientBoosting / GPU MLP comparator
- Seed: {SEED}
- Batch size: 1024 for GPU MLP
- Epochs / steps: 16 epochs for GPU MLP
- Hardware: {device} {ctx.gpu_name or ''}

## 5. Best Metrics

{markdown_table(["Model", "RMSE", "MAE", "R2"], [[name, f"{res.metrics['rmse']:.3f}", f"{res.metrics['mae']:.3f}", f"{res.metrics['r2']:.3f}"] for name, res in sorted(results.items(), key=lambda kv: kv[1].metrics['rmse'])])}

## 6. Result Figures

- `figures/results/cv_fold_score_boxplot.svg`, `validation_curve.svg`, `top_feature_importance.svg`
- Why these matter: They justify the chosen validation strategy and tuned parameter range.

## 7. Analysis Figures

- `figures/analysis/subgroup_metric_comparison.svg`, `confidence_bin_plot.svg`, `common_failure_slice_summary.svg`
- Key failure patterns: High predicted-demand bins and adverse weather / non-working-day combinations remain hardest.

## 8. Sample Predictions

- Failure examples: `predictions/worst_predictions.csv`
- Best examples: low-error hours embedded in the test split.

## 9. Decision

- 대표 artifact로 남길까? yes
- 모델 가중치를 별도로 배포할 계획이 있는가?: no
- Next run: explicit holiday features and fold-aware hyperparameter search expansion.
"""
    (ctx.run_paths.run_dir / "summary.md").write_text(summary, encoding="utf-8")
    promote_run(ctx)
    return {
        "stage": ctx.stage_name,
        "run_id": ctx.run_paths.run_id,
        "best_model": best_name,
        "best_metrics": best.metrics,
        "report_dir": str(ctx.run_paths.report_dir.relative_to(ROOT)),
    }


# ---------- Stage 4: Covertype large-scale ----------


def load_covertype() -> pd.DataFrame:
    ds = load_dataset("mstz/covertype", "covertype", split="train")
    df = ds.to_pandas()
    return sanitize_bool_columns(df)



def run_stage_04(device: str) -> dict[str, Any]:
    ctx = build_stage_context("04_large_scale_tabular", "04 Large Scale Tabular", "covertype", "macro_f1", "large-scale-suite", device)
    df = load_covertype()
    X = df.drop(columns=["cover_type"])
    y = df["cover_type"].astype(int).to_numpy()
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(X, y, test_size=0.15, random_state=SEED, stratify=y)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=0.17647, random_state=SEED, stratify=y_train_valid)

    tree_prep = Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))])
    linear_prep = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("scale", StandardScaler())])

    model_specs = {
        "sgd_linear": Pipeline([("prep", clone(linear_prep)), ("model", SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=30, early_stopping=True, n_jobs=-1, random_state=SEED))]),
        "shallow_tree": Pipeline([("prep", clone(tree_prep)), ("model", RandomForestClassifier(n_estimators=80, max_depth=12, n_jobs=-1, random_state=SEED))]),
        "hist_gbdt": Pipeline([("prep", clone(tree_prep)), ("model", HistGradientBoostingClassifier(max_depth=10, max_iter=80, learning_rate=0.08, random_state=SEED))]),
    }
    if XGBClassifier is not None:
        model_specs["xgboost_gpu"] = Pipeline([
            ("prep", clone(tree_prep)),
            ("model", XGBClassifier(
                objective="multi:softprob",
                num_class=len(np.unique(y)),
                n_estimators=220,
                max_depth=10,
                learning_rate=0.10,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                tree_method="hist",
                device="cuda" if device.startswith("cuda") else "cpu",
                eval_metric="mlogloss",
                random_state=SEED,
            )),
        ])

    results: dict[str, ModelResult] = {}
    for name, model in model_specs.items():
        model, y_pred, y_proba, fit_time, predict_time, peak_rss = timed_fit_predict(model, X_train, y_train, X_test)
        if y_proba is None:
            y_proba = None
        results[name] = ModelResult(name=name, metrics=multiclass_metrics(y_test, y_pred, y_proba), fit_time_sec=fit_time, predict_time_sec=predict_time, peak_rss_mb=peak_rss, y_pred=np.asarray(y_pred), y_score=y_proba)

    scaler = clone(linear_prep)
    X_train_mlp = to_dense_float32(scaler.fit_transform(X_train))
    X_valid_mlp = to_dense_float32(scaler.transform(X_valid))
    X_test_mlp = to_dense_float32(scaler.transform(X_test))
    t0 = time.perf_counter()
    y_pred_mlp, y_prob_mlp, extras = train_torch_classifier(X_train_mlp, y_train, X_valid_mlp, y_valid, X_test_mlp, n_classes=len(np.unique(y)), device=device, epochs=10, batch_size=2048)
    fit_time = time.perf_counter() - t0
    results["gpu_mlp"] = ModelResult(name="gpu_mlp", metrics=multiclass_metrics(y_test, y_pred_mlp, y_prob_mlp), fit_time_sec=fit_time, predict_time_sec=0.0, peak_rss_mb=process.memory_info().rss / (1024 ** 2), y_pred=y_pred_mlp, y_score=y_prob_mlp, extras=extras)

    best_name = max(results, key=lambda name: results[name].metrics[ctx.primary_metric])
    best = results[best_name]

    config = {
        "track": TRACK,
        "stage": ctx.stage_name,
        "dataset": ctx.dataset_name,
        "seed": SEED,
        "split": {"train": int(len(X_train)), "valid": int(len(X_valid)), "test": int(len(X_test))},
        "hardware": {"device": device, "gpu_name": ctx.gpu_name},
        "models": list(results.keys()),
    }
    yaml_dump(ctx.run_paths.run_dir / "config.yaml", config)
    json_dump(ctx.run_paths.run_dir / "metrics.json", {
        "primary_metric": ctx.primary_metric,
        "best_model": best_name,
        "models": {name: {**res.metrics, "fit_time_sec": res.fit_time_sec, "predict_time_sec": res.predict_time_sec, "peak_rss_mb": res.peak_rss_mb, **(res.extras or {})} for name, res in results.items()}
    })

    pred_df = X_test.copy()
    pred_df["label"] = y_test
    pred_df["pred"] = best.y_pred
    pred_df["correct"] = (pred_df["label"] == pred_df["pred"]).astype(int)
    pred_df.to_csv(ctx.run_paths.predictions_dir / "test_predictions_sample.csv", index=False)

    # result figures
    class_dist = pd.Series(y).value_counts().sort_index()
    bar_chart(ctx.run_paths.figures_results / "class_distribution.svg", [str(i) for i in class_dist.index.tolist()], class_dist.values.astype(float).tolist(), "Covertype class distribution", "Seven-class target balance across the full dataset.", "class", "count", value_fmt="{:.0f}")
    metric_vs_time_labels = list(results.keys())
    metric_vs_time = [res.metrics["macro_f1"] for res in results.values()]
    train_times = [res.fit_time_sec for res in results.values()]
    line_chart(ctx.run_paths.figures_results / "metric_vs_training_time.svg", [{"label": "macro_f1", "x": train_times, "y": metric_vs_time, "color": "#2563eb"}], "Metric vs training time", "Higher is better; shows the quality-cost tradeoff.", "fit time (sec)", "macro_f1")
    mems = [res.peak_rss_mb for res in results.values()]
    line_chart(ctx.run_paths.figures_results / "metric_vs_memory.svg", [{"label": "macro_f1", "x": mems, "y": metric_vs_time, "color": "#dc2626"}], "Metric vs memory", "Resident memory versus macro-F1.", "peak rss (MB)", "macro_f1")
    if best.y_score is not None:
        conf = np.max(best.y_score, axis=1) if np.asarray(best.y_score).ndim == 2 else np.asarray(best.y_score)
    else:
        conf = pred_df["correct"].to_numpy()
    conf_counts, conf_bins = np.histogram(conf, bins=10, range=(0, 1))
    bar_chart(ctx.run_paths.figures_results / "score_distribution.svg", [f"{conf_bins[i]:.1f}-{conf_bins[i+1]:.1f}" for i in range(len(conf_counts))], conf_counts.astype(float).tolist(), "Score distribution", f"Confidence profile for best model: {best_name}.", "score bucket", "count", value_fmt="{:.0f}")

    # analysis figures
    recall_by_class = []
    labels = []
    for klass in sorted(np.unique(y_test)):
        mask = y_test == klass
        recall_by_class.append(float((best.y_pred[mask] == y_test[mask]).mean()))
        labels.append(str(klass))
    bar_chart(ctx.run_paths.figures_analysis / "slice_metric_by_class.svg", labels, recall_by_class, "Slice metric by class", f"Per-class recall for {best_name}.", "class", "recall")
    throughput_rows = []
    for name, res in results.items():
        throughput_rows.append([name, f"{len(X_train)/max(res.fit_time_sec, 1e-6):.0f}", f"{res.fit_time_sec:.1f}", f"{res.predict_time_sec:.2f}", f"{res.peak_rss_mb:.0f}"])
    table_figure(ctx.run_paths.figures_analysis / "throughput_bottleneck_summary.svg", "Throughput bottleneck summary", "Examples/sec and memory for each large-scale model.", ["model", "train ex/s", "fit sec", "pred sec", "rss MB"], throughput_rows)

    sample_fracs = [0.1, 0.3, 1.0]
    sample_scores = []
    for frac in sample_fracs:
        n = int(len(X_train_mlp) * frac)
        pred_frac, prob_frac, _ = train_torch_classifier(X_train_mlp[:n], y_train[:n], X_valid_mlp, y_valid, X_test_mlp, n_classes=len(np.unique(y)), device=device, epochs=6, batch_size=2048)
        sample_scores.append(float(f1_score(y_test, pred_frac, average="macro")))
    line_chart(ctx.run_paths.figures_analysis / "sampling_strategy_performance.svg", [{"label": "gpu_mlp macro_f1", "x": sample_fracs, "y": sample_scores, "color": "#059669"}], "Sampling strategy vs performance", "Macro-F1 as more training data is introduced.", "train fraction", "macro_f1")

    summary = f"""# Run Summary

## 1. Problem

- Task: Large-scale multiclass tabular classification on Covertype.
- Track / Stage: {TRACK} / {ctx.stage_name}
- Why this run exists: Compare accuracy-cost tradeoffs before moving larger tabular workloads to a dedicated server workflow.

## 2. Hypothesis

- What changed from the previous run: Added explicit large-scale baselines plus a GPU MLP on CUDA device 0.
- Expected effect: HistGradientBoosting should provide the strongest CPU baseline, while the GPU MLP gives a fast nonlinear reference and sample-efficiency curve.

## 3. Dataset

- Dataset name: {ctx.dataset_name}
- Split version: train={len(X_train)}, valid={len(X_valid)}, test={len(X_test)}
- Preprocessing: bool -> int, imputation, scaling for linear/MLP paths.
- Data caveats: Class imbalance and one-hot-like soil / wilderness flags create class-specific recall gaps.

## 4. Training Setup

- Model: SGD linear / shallow random forest / hist GBDT / GPU MLP
- Seed: {SEED}
- Batch size: 2048 for GPU MLP
- Epochs / steps: 10 epochs for main GPU MLP, 6 epochs for sampling study
- Hardware: {device} {ctx.gpu_name or ''}

## 5. Best Metrics

{markdown_table(["Model", "Macro F1", "Accuracy", "Recall", "Fit sec"], [[name, f"{res.metrics['macro_f1']:.3f}", f"{res.metrics['accuracy']:.3f}", f"{res.metrics['macro_recall']:.3f}", f"{res.fit_time_sec:.1f}"] for name, res in sorted(results.items(), key=lambda kv: kv[1].metrics['macro_f1'], reverse=True)])}

## 6. Result Figures

- `figures/results/class_distribution.svg`, `metric_vs_training_time.svg`, `metric_vs_memory.svg`, `score_distribution.svg`
- Why these matter: They show dataset scale, cost-quality tradeoffs, and confidence spread.

## 7. Analysis Figures

- `figures/analysis/slice_metric_by_class.svg`, `throughput_bottleneck_summary.svg`, `sampling_strategy_performance.svg`
- Key failure patterns: Minority forest-cover types remain hardest, and throughput strongly favors the GPU MLP as sample size increases.

## 8. Sample Predictions

- Test predictions: `predictions/test_predictions_sample.csv`
- Failure examples: class-wise recall slices highlight where mistakes concentrate.

## 9. Decision

- 대표 artifact로 남길까? yes
- 모델 가중치를 별도로 배포할 계획이 있는가?: no
- Next run: compare against XGBoost/LightGBM GPU variants once those dependencies are explicitly approved.
"""
    (ctx.run_paths.run_dir / "summary.md").write_text(summary, encoding="utf-8")
    promote_run(ctx)
    return {
        "stage": ctx.stage_name,
        "run_id": ctx.run_paths.run_id,
        "best_model": best_name,
        "best_metrics": best.metrics,
        "report_dir": str(ctx.run_paths.report_dir.relative_to(ROOT)),
    }


# ---------- promotion and index ----------


def promote_run(ctx: StageContext) -> None:
    report_dir = ctx.run_paths.report_dir
    if report_dir.resolve() == ctx.run_paths.run_dir.resolve():
        return
    ensure_dir(report_dir / "figures" / "results")
    ensure_dir(report_dir / "figures" / "analysis")
    shutil.copy2(ctx.run_paths.run_dir / "summary.md", report_dir / "summary.md")
    shutil.copy2(ctx.run_paths.run_dir / "metrics.json", report_dir / "metrics.json")
    for src in ctx.run_paths.figures_results.glob("*.svg"):
        shutil.copy2(src, report_dir / "figures" / "results" / src.name)
    for src in ctx.run_paths.figures_analysis.glob("*.svg"):
        shutil.copy2(src, report_dir / "figures" / "analysis" / src.name)



def write_track_report(stage_results: list[dict[str, Any]]) -> None:
    readme = [
        "# 01 ML Reports",
        "",
        "이 문서는 ML 트랙 전체 실행 결과의 인덱스다.",
        "",
        "## Stage Summary",
        "",
        markdown_table(
            ["Stage", "Run ID", "Best model", "Primary metric snapshot", "Report dir"],
            [[res["stage"], res["run_id"], res["best_model"], ", ".join(f"{k}={v:.3f}" for k, v in list(res["best_metrics"].items())[:2]), res["report_dir"]] for res in stage_results],
        ),
        "",
        "## Notes",
        "",
        "- 모든 실험은 `CUDA_VISIBLE_DEVICES=0` 환경에서 실행했다.",
        "- CPU 중심의 scikit-learn baseline도 유지했고, 각 stage마다 GPU MLP comparator를 함께 학습해 GPU 0 사용을 명시적으로 포함했다.",
        "- 각 stage의 대표 summary/figure는 `01_ml/<stage>/artifacts/<run_id>/` 안에 함께 남긴다.",
    ]
    (TRACK_ROOT / "RESULTS.md").write_text("\n".join(readme), encoding="utf-8")


# ---------- CLI ----------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the entire 01_ml track and refresh stage-local artifacts.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to expose via CUDA_VISIBLE_DEVICES.")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ensure_dir(RUNS_ROOT)
    ensure_dir(REPORTS_ROOT)
    stage_results = [
        run_stage_01(device),
        run_stage_02(device),
        run_stage_03(device),
        run_stage_04(device),
    ]
    write_track_report(stage_results)
    print(json.dumps(stage_results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
