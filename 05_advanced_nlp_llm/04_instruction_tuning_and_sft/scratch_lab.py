from __future__ import annotations

import json
import re
from pathlib import Path
from xml.sax.saxutils import escape

UNIT_ROOT = Path(__file__).resolve().parent
OUT_DIR = UNIT_ROOT / 'artifacts' / 'scratch-manual'
METRICS_PATH = OUT_DIR / 'metrics.json'
FIGURE_PATH = OUT_DIR / 'sft_template_loss.svg'
TOKEN_RE = re.compile(r'[A-Za-z0-9_<>|/#]+|[가-힣]+|[0-9]+|[^\s]')

SYSTEM = '너는 한국어로 짧고 정확하게 답하는 튜터다.'
USER = 'instruction format과 SFT의 차이를 한 문장으로 설명하라.'
ASSISTANT = 'instruction format은 요청과 답변 경계를 정하고, SFT는 그 형식의 정답 응답을 모방하도록 학습한다.'
PLAIN_PREFIX = '### Instruction:'
PLAIN_RESPONSE = '### Response:'
CHAT_SYSTEM = '<|system|>'
CHAT_USER = '<|user|>'
CHAT_ASSISTANT = '<|assistant|>'
EOS = '<|eos|>'


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def _round(value: float) -> float:
    return round(value, 6)


def _template_view_plain() -> dict[str, object]:
    prompt = f'{PLAIN_PREFIX}\n{USER}\n{PLAIN_RESPONSE}\n'
    response = f'{ASSISTANT} {EOS}'
    serialized = prompt + response
    prompt_tokens = _tokens(prompt)
    assistant_tokens = _tokens(response)
    return {
        'name': 'plain_instruction',
        'roles': ['instruction', 'response'],
        'serialized_prefix': serialized[:80],
        'target_region': 'assistant_response_only',
        'prompt_tokens': len(prompt_tokens),
        'assistant_tokens': len(assistant_tokens),
        'total_tokens': len(prompt_tokens) + len(assistant_tokens),
        'loss_tokens': len(assistant_tokens),
        'ignored_prompt_tokens': len(prompt_tokens),
    }


def _template_view_chat() -> dict[str, object]:
    prompt = f'{CHAT_SYSTEM}\n{SYSTEM}\n{CHAT_USER}\n{USER}\n{CHAT_ASSISTANT}\n'
    response = f'{ASSISTANT} {EOS}'
    serialized = prompt + response
    prompt_tokens = _tokens(prompt)
    assistant_tokens = _tokens(response)
    return {
        'name': 'chat_template',
        'roles': ['system', 'user', 'assistant'],
        'serialized_prefix': serialized[:90],
        'target_region': 'assistant_response_only',
        'prompt_tokens': len(prompt_tokens),
        'assistant_tokens': len(assistant_tokens),
        'total_tokens': len(prompt_tokens) + len(assistant_tokens),
        'loss_tokens': len(assistant_tokens),
        'ignored_prompt_tokens': len(prompt_tokens),
        'role_boundaries': {
            'system_start': CHAT_SYSTEM,
            'user_start': CHAT_USER,
            'assistant_start': CHAT_ASSISTANT,
            'stop_marker': EOS,
        },
    }


def _role_framing() -> dict[str, object]:
    with_system = {
        'answer': '간결한 한국어 튜터 답변: instruction format은 경계이고 SFT는 모방 학습이다.',
        'tone_constraint_score': 0.86,
        'constraint_hits': ['한국어', '간결', '튜터'],
    }
    without_system = {
        'answer': 'SFT는 데이터를 따라 학습합니다. 여러 설명이 가능하며 자세한 배경도 있습니다.',
        'tone_constraint_score': 0.52,
        'constraint_hits': ['한국어'],
    }
    return {
        'with_system_message': with_system,
        'without_system_message': without_system,
        'system_constraint_delta': _round(with_system['tone_constraint_score'] - without_system['tone_constraint_score']),
        'recommended_for_role_control': 'chat_template',
        'lesson': 'system/user/assistant role tags are conditioning tokens, not external metadata.',
    }


def _imitation_vs_helpfulness() -> dict[str, object]:
    return {
        'format_imitation_score': 0.91,
        'helpfulness_proxy_score': 0.74,
        'canned_response_risk': 0.28,
        'examples': {
            'imitated_reference': '물론입니다. 아래와 같이 답할 수 있습니다...',
            'more_helpful_revision': '요청의 핵심을 한 문장으로 답하고, 필요한 경우에만 예시를 추가한다.',
        },
        'tradeoff_note': 'SFT can copy reference style faster than it learns which answer a human would prefer.',
    }


def _render_svg(template_views: dict[str, dict[str, object]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    width = 720
    height = 380
    left = 90
    top = 86
    bar_width = 68
    gap = 130
    chart_height = 180
    max_tokens = max(int(view['total_tokens']) for view in template_views.values()) + 8

    def y_for_tokens(tokens: int) -> float:
        return top + chart_height - (tokens / max_tokens) * chart_height

    grid = []
    for tick in range(0, max_tokens + 1, 10):
        y = y_for_tokens(tick)
        grid.append(f'<line x1="{left - 20}" y1="{y:.1f}" x2="620" y2="{y:.1f}" stroke="#e1e8f3" />')
        grid.append(f'<text x="{left - 28}" y="{y + 4:.1f}" font-size="12" text-anchor="end" fill="#53657d">{tick}</text>')

    bars = []
    for index, (name, view) in enumerate(template_views.items()):
        x = left + index * (bar_width * 2 + gap)
        prompt = int(view['prompt_tokens'])
        assistant = int(view['assistant_tokens'])
        prompt_y = y_for_tokens(prompt)
        assistant_y = y_for_tokens(assistant)
        total_y = y_for_tokens(prompt + assistant)
        bars.append(f'<rect x="{x}" y="{prompt_y:.1f}" width="{bar_width}" height="{top + chart_height - prompt_y:.1f}" fill="#d7e3f7" />')
        bars.append(f'<rect x="{x + bar_width + 12}" y="{assistant_y:.1f}" width="{bar_width}" height="{top + chart_height - assistant_y:.1f}" fill="#4f7cff" />')
        bars.append(f'<line x1="{x - 8}" y1="{total_y:.1f}" x2="{x + bar_width * 2 + 20}" y2="{total_y:.1f}" stroke="#263a52" stroke-dasharray="4 4" />')
        bars.append(f'<text x="{x + bar_width}" y="{top + chart_height + 24}" font-size="13" text-anchor="middle" fill="#253b55">{escape(name)}</text>')
        bars.append(f'<text x="{x + bar_width / 2}" y="{prompt_y - 6:.1f}" font-size="12" text-anchor="middle" fill="#44566f">prompt {prompt}</text>')
        bars.append(f'<text x="{x + bar_width + 12 + bar_width / 2}" y="{assistant_y - 6:.1f}" font-size="12" text-anchor="middle" fill="#143e9f">assistant {assistant}</text>')

    legend = (
        '<rect x="92" y="322" width="18" height="12" fill="#d7e3f7" />'
        '<text x="118" y="333" font-size="13" fill="#263a52">Prompt tokens masked out</text>'
        '<rect x="306" y="322" width="18" height="12" fill="#4f7cff" />'
        '<text x="332" y="333" font-size="13" fill="#263a52">Assistant loss tokens</text>'
        '<line x1="506" y1="328" x2="540" y2="328" stroke="#263a52" stroke-dasharray="4 4" />'
        '<text x="548" y="333" font-size="13" fill="#263a52">Full sequence tokens</text>'
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#f8fbff" />'
        '<text x="28" y="36" font-size="24" font-weight="bold" fill="#10203a">Instruction tuning and SFT</text>'
        '<text x="28" y="62" font-size="14" fill="#42556d">Assistant-only loss mask over toy input-output templates</text>'
        f'<text x="{left - 20}" y="{top - 20}" font-size="13" fill="#42556d">token count</text>'
        + ''.join(grid)
        + ''.join(bars)
        + legend
        + '</svg>'
    )
    FIGURE_PATH.write_text(svg, encoding='utf-8')


def run() -> None:
    template_views = {
        'plain_instruction': _template_view_plain(),
        'chat_template': _template_view_chat(),
    }
    _render_svg(template_views)
    prompt_tokens = sum(int(view['prompt_tokens']) for view in template_views.values())
    assistant_tokens = sum(int(view['assistant_tokens']) for view in template_views.values())
    total_tokens = sum(int(view['total_tokens']) for view in template_views.values())
    metrics = {
        'setup': {
            'unit': '04_instruction_tuning_and_sft',
            'mode': 'cpu_safe_deterministic_toy_instruction_tuning',
            'example_count': 1,
        },
        'template_views': template_views,
        'loss_masking': {
            'target_region': 'assistant_response_only',
            'prompt_tokens_masked_out': prompt_tokens,
            'assistant_loss_tokens': assistant_tokens,
            'full_sequence_loss_tokens': total_tokens,
            'assistant_loss_share': _round(assistant_tokens / total_tokens),
            'ignored_label': -100,
        },
        'role_framing': _role_framing(),
        'imitation_vs_helpfulness': _imitation_vs_helpfulness(),
        'figure_path': 'artifacts/scratch-manual/sft_template_loss.svg',
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    summary = {
        'setup': metrics['setup'],
        'loss_masking': metrics['loss_masking'],
        'role_framing': {
            'recommended_for_role_control': metrics['role_framing']['recommended_for_role_control'],
            'system_constraint_delta': metrics['role_framing']['system_constraint_delta'],
        },
        'figure_path': metrics['figure_path'],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == '__main__':
    run()
