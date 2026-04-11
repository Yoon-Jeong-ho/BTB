from __future__ import annotations

import json
import math
import re
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
OUT_DIR = UNIT_ROOT / 'artifacts' / 'framework-manual'
METRICS_PATH = OUT_DIR / 'metrics.json'
TOKEN_RE = re.compile(r'[A-Za-z0-9_<>|/#]+|[가-힣]+|[0-9]+|[^\s]')
PAD = '<pad>'
EOS = '<|eos|>'
IGNORED_LABEL = -100
DATASET = [
    {
        'system': '한국어로 간결하게 답하는 튜터다.',
        'user': 'instruction format이 무엇인가?',
        'assistant': '요청과 응답 경계를 정해 모델이 assistant 답변 위치를 알게 하는 형식이다.',
    },
    {
        'system': '한 문장으로 답하라.',
        'user': 'SFT는 무엇을 학습하는가?',
        'assistant': 'SFT는 reference assistant answer를 supervised fine-tuning으로 모방하게 학습한다.',
    },
    {
        'system': '역할을 구분해서 설명하라.',
        'user': 'system/user/assistant tag는 왜 필요한가?',
        'assistant': '각 role tag는 제약, 요청, 생성 target을 나누는 conditioning signal이다.',
    },
    {
        'system': 'tradeoff를 포함하라.',
        'user': 'imitation과 helpfulness는 왜 다른가?',
        'assistant': '모방은 reference 형식을 따르는 것이고 helpfulness는 사용자 목적에 더 맞는 답을 고르는 것이다.',
    },
]


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def _serialize(example: dict[str, str]) -> tuple[list[str], list[str]]:
    prompt = (
        f'<|system|> {example["system"]} '
        f'<|user|> {example["user"]} '
        '<|assistant|> '
    )
    response = f'{example["assistant"]} {EOS}'
    return _tokens(prompt), _tokens(response)


def _build_vocab(serialized: list[tuple[list[str], list[str]]]) -> dict[str, int]:
    vocab = {PAD: 0}
    for prompt_tokens, assistant_tokens in serialized:
        for token in prompt_tokens + assistant_tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def _make_batch() -> dict[str, object]:
    serialized = [_serialize(example) for example in DATASET]
    vocab = _build_vocab(serialized)
    max_len = max(len(prompt) + len(answer) for prompt, answer in serialized)
    input_ids: list[list[int]] = []
    labels: list[list[int]] = []
    loss_mask: list[list[int]] = []
    prompt_count = 0
    assistant_count = 0
    previews: list[dict[str, object]] = []
    for index, (prompt_tokens, assistant_tokens) in enumerate(serialized):
        ids = [vocab[token] for token in prompt_tokens + assistant_tokens]
        label_row = [IGNORED_LABEL] * len(prompt_tokens) + [vocab[token] for token in assistant_tokens]
        mask_row = [0] * len(prompt_tokens) + [1] * len(assistant_tokens)
        pad = max_len - len(ids)
        input_ids.append(ids + [vocab[PAD]] * pad)
        labels.append(label_row + [IGNORED_LABEL] * pad)
        loss_mask.append(mask_row + [0] * pad)
        prompt_count += len(prompt_tokens)
        assistant_count += len(assistant_tokens)
        previews.append(
            {
                'example_index': index,
                'prompt_tokens': len(prompt_tokens),
                'assistant_tokens': len(assistant_tokens),
                'first_prompt_tokens': prompt_tokens[:5],
                'first_assistant_tokens': assistant_tokens[:5],
            }
        )
    return {
        'vocab': vocab,
        'input_ids': input_ids,
        'labels': labels,
        'loss_mask': loss_mask,
        'max_len': max_len,
        'prompt_count': prompt_count,
        'assistant_count': assistant_count,
        'previews': previews,
    }


def _training_curve(vocab_size: int, assistant_tokens: int) -> list[dict[str, float | int]]:
    initial_loss = math.log(vocab_size)
    curve = []
    for epoch in range(6):
        assistant_loss = initial_loss * math.exp(-0.22 * epoch) + 0.018 * (assistant_tokens / 100)
        template_adherence = min(0.96, 0.43 + 0.095 * epoch)
        helpfulness_proxy = min(0.81, 0.57 + 0.041 * epoch)
        curve.append(
            {
                'epoch': epoch,
                'assistant_loss': round(assistant_loss, 6),
                'template_adherence': round(template_adherence, 6),
                'helpfulness_proxy': round(helpfulness_proxy, 6),
            }
        )
    return curve


def run() -> None:
    batch = _make_batch()
    input_ids = batch['input_ids']
    labels = batch['labels']
    loss_mask = batch['loss_mask']
    vocab = batch['vocab']
    max_len = int(batch['max_len'])
    prompt_count = int(batch['prompt_count'])
    assistant_count = int(batch['assistant_count'])
    curve = _training_curve(len(vocab), assistant_count)
    final = curve[-1]
    metrics = {
        'device': 'cpu',
        'framework': 'deterministic_numeric_sft',
        'dataset_size': len(DATASET),
        'vocab_size': len(vocab),
        'max_sequence_length': max_len,
        'batch_shape': {
            'input_ids': [len(input_ids), max_len],
            'labels': [len(labels), max_len],
            'loss_mask': [len(loss_mask), max_len],
        },
        'loss_mask_summary': {
            'prompt_loss_tokens': 0,
            'assistant_loss_tokens': assistant_count,
            'masked_prompt_tokens': prompt_count,
            'ignored_label': IGNORED_LABEL,
            'assistant_loss_share': round(assistant_count / (prompt_count + assistant_count), 6),
        },
        'template': {
            'name': 'chat_template',
            'role_order': ['system', 'user', 'assistant'],
            'objective': 'next_token_loss_on_assistant_tokens',
            'eos_token': EOS,
        },
        'batch_preview': batch['previews'],
        'training_curve': curve,
        'imitation_vs_helpfulness': {
            'format_imitation_final': final['template_adherence'],
            'helpfulness_proxy_final': final['helpfulness_proxy'],
            'over_imitation_risk': round(float(final['template_adherence']) - float(final['helpfulness_proxy']), 6),
            'note': 'Numeric toy SFT improves template imitation faster than helpfulness proxy.',
        },
        'next_step': {
            'why_sft_is_not_enough': 'preference_optimization_needed',
            'reason': 'SFT sees one reference answer per prompt and does not directly rank two plausible answers.',
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    summary = {
        'device': metrics['device'],
        'framework': metrics['framework'],
        'dataset_size': metrics['dataset_size'],
        'batch_shape': metrics['batch_shape'],
        'next_step': metrics['next_step'],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == '__main__':
    run()
