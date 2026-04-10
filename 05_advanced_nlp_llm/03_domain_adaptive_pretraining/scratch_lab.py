from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'scratch-manual'
FIGURE_PATH = ARTIFACT_DIR / 'dapt_tradeoff.svg'
STEPS = [0, 1, 2, 3, 4, 5, 6]
BASE_DOMAIN_LOSS = 2.74
BASE_GENERAL_LOSS = 1.82
GENERAL_GUARDRAIL = 0.18


def _round(value: float) -> float:
    return round(value, 6)


def _first_exceeded(losses: list[float], baseline: float, guardrail: float) -> int:
    for step, loss in zip(STEPS, losses):
        if _round(loss - baseline) > guardrail:
            return step
    return 0


def _best_stop_step(domain_losses: list[float], general_losses: list[float]) -> int:
    candidates: list[tuple[float, int]] = []
    for step, domain_loss, general_loss in zip(STEPS, domain_losses, general_losses):
        if _round(general_loss - BASE_GENERAL_LOSS) <= GENERAL_GUARDRAIL:
            candidates.append((domain_loss, step))
    if not candidates:
        return 0
    return min(candidates)[1]


def _strategy(
    *,
    domain_share: float,
    general_share: float,
    domain_losses: list[float],
    general_losses: list[float],
    note: str,
) -> dict[str, object]:
    final_domain = domain_losses[-1]
    final_general = general_losses[-1]
    stop_step = _best_stop_step(domain_losses, general_losses)
    return {
        'domain_share': domain_share,
        'general_share': general_share,
        'steps': STEPS,
        'domain_val_loss_by_step': domain_losses,
        'general_val_loss_by_step': general_losses,
        'in_domain_gain_final': _round(BASE_DOMAIN_LOSS - final_domain),
        'general_regression_final': _round(final_general - BASE_GENERAL_LOSS),
        'recommended_stop_step': stop_step,
        'guardrail_exceeded_step': _first_exceeded(
            general_losses,
            BASE_GENERAL_LOSS,
            GENERAL_GUARDRAIL,
        ),
        'note': note,
    }


def _render_svg(strategies: dict[str, dict[str, object]]) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    width = 720
    height = 420
    left = 72
    top = 78
    chart_width = 560
    chart_height = 230
    y_min = 1.75
    y_max = 2.8
    colors = {
        ('pure_domain', 'domain'): '#4f7cff',
        ('pure_domain', 'general'): '#d64545',
        ('replay_mixture', 'domain'): '#34a853',
        ('replay_mixture', 'general'): '#f2994a',
    }

    def x_for_step(step: int) -> float:
        return left + (step / max(STEPS)) * chart_width

    def y_for_loss(loss: float) -> float:
        scaled = (loss - y_min) / (y_max - y_min)
        return top + chart_height - scaled * chart_height

    grid: list[str] = []
    for tick in range(6):
        value = y_min + tick * ((y_max - y_min) / 5)
        y = y_for_loss(value)
        grid.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_width}" y2="{y:.1f}" stroke="#d8e1ef" />')
        grid.append(f'<text x="{left - 12}" y="{y + 4:.1f}" font-size="12" text-anchor="end" fill="#4d6078">{value:.2f}</text>')
    for step in STEPS:
        x = x_for_step(step)
        grid.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + chart_height}" stroke="#eef3fb" />')
        grid.append(f'<text x="{x:.1f}" y="{top + chart_height + 22}" font-size="12" text-anchor="middle" fill="#4d6078">{step}</text>')

    lines: list[str] = []
    for name, strategy in strategies.items():
        domain_losses = strategy['domain_val_loss_by_step']
        general_losses = strategy['general_val_loss_by_step']
        assert isinstance(domain_losses, list)
        assert isinstance(general_losses, list)
        for kind, losses in [('domain', domain_losses), ('general', general_losses)]:
            points = ' '.join(
                f'{x_for_step(step):.1f},{y_for_loss(float(loss)):.1f}'
                for step, loss in zip(STEPS, losses)
            )
            color = colors[(name, kind)]
            dash = ' stroke-dasharray="6 5"' if kind == 'general' else ''
            lines.append(
                f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"{dash} />'
            )
            for step, loss in zip(STEPS, losses):
                lines.append(
                    f'<circle cx="{x_for_step(step):.1f}" cy="{y_for_loss(float(loss)):.1f}" r="4" fill="{color}" />'
                )

    legend = [
        ('pure domain / in-domain', '#4f7cff', False),
        ('pure domain / General retention', '#d64545', True),
        ('replay mixture / in-domain', '#34a853', False),
        ('replay mixture / General retention', '#f2994a', True),
    ]
    legend_items: list[str] = []
    for index, (label, color, dashed) in enumerate(legend):
        x = 84 + (index % 2) * 280
        y = 342 + (index // 2) * 24
        dash = ' stroke-dasharray="6 5"' if dashed else ''
        legend_items.append(f'<line x1="{x}" y1="{y}" x2="{x + 32}" y2="{y}" stroke="{color}" stroke-width="3"{dash} />')
        legend_items.append(f'<text x="{x + 42}" y="{y + 5}" font-size="13" fill="#263a52">{escape(label)}</text>')

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#f8fbff" />'
        '<text x="28" y="34" font-size="24" font-weight="bold" fill="#10203a">Domain-adaptive pretraining trade-offs</text>'
        '<text x="28" y="58" font-size="14" fill="#42556d">Toy continued pretraining: in-domain adaptation vs General retention</text>'
        f'<text x="{left}" y="{top - 16}" font-size="14" font-weight="bold" fill="#10203a">Validation loss</text>'
        + ''.join(grid)
        + ''.join(lines)
        + f'<text x="{left + chart_width / 2}" y="{top + chart_height + 48}" font-size="13" text-anchor="middle" fill="#42556d">continued-pretraining step</text>'
        + ''.join(legend_items)
        + '</svg>'
    )
    FIGURE_PATH.write_text(svg, encoding='utf-8')


def run() -> None:
    strategies = {
        'pure_domain': _strategy(
            domain_share=1.0,
            general_share=0.0,
            domain_losses=[2.74, 2.43, 2.24, 2.11, 2.03, 2.00, 1.98],
            general_losses=[1.82, 1.88, 1.96, 2.05, 2.16, 2.27, 2.36],
            note='도메인 적응은 가장 빠르지만 general retention guardrail을 이르게 넘는다.',
        ),
        'replay_mixture': _strategy(
            domain_share=0.7,
            general_share=0.3,
            domain_losses=[2.74, 2.52, 2.37, 2.26, 2.18, 2.13, 2.10],
            general_losses=[1.82, 1.84, 1.86, 1.88, 1.91, 1.94, 1.96],
            note='general replay를 섞어 적응은 느리지만 forgetting 상승이 완만하다.',
        ),
    }
    _render_svg(strategies)

    metrics = {
        'setup': {
            'unit': '03_domain_adaptive_pretraining',
            'objective_kept_constant': 'causal_lm',
            'toy_steps': STEPS,
            'baseline_losses': {
                'domain': BASE_DOMAIN_LOSS,
                'general': BASE_GENERAL_LOSS,
            },
            'general_regression_guardrail': GENERAL_GUARDRAIL,
        },
        'domain_shift': {
            'general_terms': ['회의', '문서', '일정', '공유', '검토'],
            'domain_terms': ['환자', '혈압', '투약', '진단', '처방'],
            'term_overlap_ratio': 0.0,
            'style_shift': '짧은 업무 문서에서 구조화된 임상 기록으로 이동',
            'why_dapt': '같은 causal LM objective라도 token distribution과 문서 형식이 바뀌면 validation loss가 갈라진다.',
        },
        'data_selection': {
            'curated_domain': {
                'document_count': 4,
                'duplicate_rate': 0.0,
                'target_distribution_match': 0.92,
                'contamination_risk': 0.05,
                'selection_score': 0.87,
            },
            'noisy_large': {
                'document_count': 12,
                'duplicate_rate': 0.42,
                'target_distribution_match': 0.61,
                'contamination_risk': 0.18,
                'selection_score': 0.48,
            },
            'preferred': 'curated_domain',
            'lesson': 'DAPT에서는 문서 수보다 품질, 중복, 오염 위험, 목표 분포 적합도를 먼저 본다.',
        },
        'strategies': strategies,
        'comparison': {
            'fastest_adapter': 'pure_domain',
            'safer_retention': 'replay_mixture',
            'balanced_recommendation': 'replay_mixture',
            'stopping_note': 'pure-domain은 step 2 이후 guardrail 위험이 커지고, replay mixture는 더 늦게 멈출 여지가 있다.',
        },
        'figure_path': 'artifacts/scratch-manual/dapt_tradeoff.svg',
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / 'metrics.json').write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
