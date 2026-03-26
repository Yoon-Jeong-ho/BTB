from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TRACK = "01_ml"
RUNS_ROOT = ROOT / "runs" / TRACK
REPORTS_ROOT = ROOT / "reports" / TRACK

STAGE_META = {
    "01 Tabular Classification": {
        "dir": "01_tabular_classification",
        "title_ko": "01. 표형 분류",
        "task_ko": "Adult Census Income 이진 분류",
        "primary": "AUPRC",
        "best_keys": ["auprc", "auroc", "f1", "accuracy"],
        "result_preview": ["pr_curve.svg", "roc_curve.svg", "confusion_matrix.svg"],
        "analysis_preview": ["permutation_importance.svg", "error_slice_by_sex.svg", "confidence_vs_correctness.svg"],
        "insight": "랜덤 포레스트가 분류 순위 품질(AUPRC)에서 가장 안정적이었고, 성별 slice에서 오차율 차이가 남았다.",
    },
    "02 Tabular Regression": {
        "dir": "02_tabular_regression",
        "title_ko": "02. 표형 회귀",
        "task_ko": "California Housing 회귀",
        "primary": "RMSE",
        "best_keys": ["rmse", "mae", "r2"],
        "result_preview": ["parity_plot.svg", "residual_vs_target.svg", "learning_curve.svg"],
        "analysis_preview": ["feature_importance.svg", "regional_error_slice.svg", "worst_prediction_cases.svg"],
        "insight": "HistGradientBoostingRegressor가 가장 낮은 RMSE를 기록했고, 고가 주택/특정 지역에서 residual이 커졌다.",
    },
    "03 Model Selection And Interpretation": {
        "dir": "03_model_selection_and_interpretation",
        "title_ko": "03. 모델 선택과 해석",
        "task_ko": "Bike Sharing 시계열성 count 회귀",
        "primary": "RMSE",
        "best_keys": ["rmse", "mae", "r2"],
        "result_preview": ["cv_fold_score_boxplot.svg", "validation_curve.svg", "top_feature_importance.svg"],
        "analysis_preview": ["subgroup_metric_comparison.svg", "confidence_bin_plot.svg", "common_failure_slice_summary.svg"],
        "insight": "시간축을 보존한 CV와 tuned HGBDT가 강력했고, 악천후/비근무일 조합이 가장 어려운 slice였다.",
    },
    "04 Large Scale Tabular": {
        "dir": "04_large_scale_tabular",
        "title_ko": "04. 대규모 표형 데이터",
        "task_ko": "Covertype 대규모 다중분류",
        "primary": "Macro-F1",
        "best_keys": ["macro_f1", "accuracy", "macro_recall", "mean_confidence"],
        "result_preview": ["metric_vs_training_time.svg", "metric_vs_memory.svg", "score_distribution.svg"],
        "analysis_preview": ["slice_metric_by_class.svg", "throughput_bottleneck_summary.svg", "sampling_strategy_performance.svg"],
        "insight": "대규모 데이터에서는 비용 대비 성능 비교가 핵심이었고, class별 recall 편차를 함께 봐야 했다.",
    },
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))



def _metric_rows(models: dict[str, dict[str, Any]], ordered_keys: list[str]) -> str:
    header = ["모델"] + [k.upper() for k in ordered_keys] + ["FIT_SEC"]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    def sort_key(item: tuple[str, dict[str, Any]]) -> tuple:
        name, metrics = item
        first = metrics.get(ordered_keys[0], 0)
        if ordered_keys[0] in {"rmse", "mae"}:
            first = -first
        return (first,)
    items = sorted(models.items(), key=sort_key, reverse=True)
    for name, metrics in items:
        row = [name]
        for key in ordered_keys:
            value = metrics.get(key)
            row.append(f"{value:.4f}" if isinstance(value, (int, float)) else "-")
        fit_sec = metrics.get("fit_time_sec")
        row.append(f"{fit_sec:.2f}" if isinstance(fit_sec, (int, float)) else "-")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)



def _best_metric_line(metrics: dict[str, Any], keys: list[str]) -> str:
    parts = []
    for key in keys:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            parts.append(f"`{key}`={value:.4f}")
    return ", ".join(parts)



def _embed_images(base_dir: Path, subdir: str, names: list[str]) -> str:
    chunks = []
    for name in names:
        path = base_dir / "figures" / subdir / name
        if path.exists():
            chunks.append(f"### {name}\n\n![](figures/{subdir}/{name})\n")
    return "\n".join(chunks)



def _hero_image_rel(meta: dict[str, Any], report_dir: Path) -> str | None:
    for name in meta["result_preview"]:
        path = report_dir / "figures" / "results" / name
        if path.exists():
            return f"{report_dir.relative_to(REPORTS_ROOT).as_posix()}/figures/results/{name}"
    return None



def build_stage_summary_ko(stage_result: dict[str, Any]) -> str:
    meta = STAGE_META[stage_result["stage"]]
    report_dir = ROOT / stage_result["report_dir"]
    metrics = _load_json(report_dir / "metrics.json")
    best_model = metrics["best_model"]
    best_metrics = metrics["models"][best_model]
    model_table = _metric_rows(metrics["models"], meta["best_keys"])
    return f"""# {meta['title_ko']} 결과 요약

## 한 줄 결론

- 과제: {meta['task_ko']}
- 최고 모델: `{best_model}`
- 핵심 지표: {_best_metric_line(best_metrics, meta['best_keys'])}
- 해석: {meta['insight']}

## 모델 비교

{model_table}

## 결과 Figure

{_embed_images(report_dir, 'results', meta['result_preview'])}

## 분석 Figure

{_embed_images(report_dir, 'analysis', meta['analysis_preview'])}

## 다음 액션

- 최고 점수만 보지 말고, figure에서 드러난 실패 slice를 다음 실험 가설로 연결한다.
- 원시 산출물은 `runs/{TRACK}/{meta['dir']}/{stage_result['run_id']}/` 아래에서 확인할 수 있다.
"""



def localize_ml_reports(stage_results: list[dict[str, Any]]) -> None:
    for result in stage_results:
        meta = STAGE_META[result["stage"]]
        run_dir = RUNS_ROOT / meta["dir"] / result["run_id"]
        report_dir = ROOT / result["report_dir"]
        summary_ko = build_stage_summary_ko(result)
        (run_dir / "summary.md").write_text(summary_ko, encoding="utf-8")
        (report_dir / "summary.md").write_text(summary_ko, encoding="utf-8")
        (report_dir / "README.md").write_text(summary_ko, encoding="utf-8")



def write_track_index_ko(stage_results: list[dict[str, Any]]) -> None:
    rows = []
    preview_blocks = []
    for result in stage_results:
        meta = STAGE_META[result["stage"]]
        report_dir = ROOT / result["report_dir"]
        metrics = _load_json(report_dir / "metrics.json")
        best_model = metrics["best_model"]
        best_metrics = metrics["models"][best_model]
        rows.append(
            f"| {meta['title_ko']} | `{best_model}` | {_best_metric_line(best_metrics, meta['best_keys'][:2])} | [{result['run_id']}]({report_dir.relative_to(REPORTS_ROOT).as_posix()}/README.md) |"
        )
        hero = _hero_image_rel(meta, report_dir)
        if hero:
            preview_blocks.append(
                f"## {meta['title_ko']}\n\n- 과제: {meta['task_ko']}\n- 요약: {meta['insight']}\n\n![]({hero})\n"
            )
    body = [
        "# 01 ML 리포트 인덱스",
        "",
        "ML 트랙 전체 실행 결과를 한 눈에 보기 위한 요약 문서다.",
        "",
        "## 실행 환경",
        "",
        "- 전용 conda 환경: [`01_ml/env/README.md`](../../01_ml/env/README.md)",
        "- 실행 명령: `CUDA_VISIBLE_DEVICES=0 conda run -n btb-01-ml python scripts/01_ml/run_all.py --gpu 0`",
        "- 원시 산출물: `runs/01_ml/...`",
        "",
        "## Stage 요약표",
        "",
        "| Stage | 최고 모델 | 핵심 지표 | 링크 |",
        "| --- | --- | --- | --- |",
        *rows,
        "",
        *preview_blocks,
    ]
    (REPORTS_ROOT / "README.md").write_text("\n".join(body), encoding="utf-8")
