from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.device_runtime import resolve_torch_device

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


class TinyNextTokenModel(torch.nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 48) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.output = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.output(torch.tanh(self.embedding(input_ids)))


def _assistant_loss(model: TinyNextTokenModel, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = model(input_ids[:, :-1])
    shifted_labels = labels[:, 1:]
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        shifted_labels.reshape(-1),
        ignore_index=IGNORED_LABEL,
    )


def _curve_row(epoch: int, loss: float, initial_loss: float) -> dict[str, float | int]:
    learned_share = max(0.0, min(0.99, 1.0 - (loss / initial_loss)))
    helpfulness_proxy = learned_share * 0.8
    return {
        'epoch': epoch,
        'assistant_loss': round(loss, 6),
        'template_adherence': round(learned_share, 6),
        'helpfulness_proxy': round(helpfulness_proxy, 6),
    }


def run() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(7)
    device = resolve_torch_device()
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(7)

    batch = _make_batch()
    input_ids = batch['input_ids']
    labels = batch['labels']
    loss_mask = batch['loss_mask']
    vocab = batch['vocab']
    max_len = int(batch['max_len'])
    prompt_count = int(batch['prompt_count'])
    assistant_count = int(batch['assistant_count'])
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)
    label_tensor = torch.tensor(labels, dtype=torch.long, device=device)
    model = TinyNextTokenModel(len(vocab)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.08, weight_decay=0.0)

    with torch.no_grad():
        initial_loss = float(_assistant_loss(model, input_tensor, label_tensor).detach().cpu())
    curve = [_curve_row(0, initial_loss, initial_loss)]
    for epoch in range(1, 61):
        optimizer.zero_grad()
        loss = _assistant_loss(model, input_tensor, label_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            curve.append(_curve_row(epoch, float(loss.detach().cpu()), initial_loss))

    with torch.no_grad():
        final_loss = float(_assistant_loss(model, input_tensor, label_tensor).detach().cpu())
    curve[-1] = _curve_row(60, final_loss, initial_loss)
    final = curve[-1]
    metrics = {
        'device': device.type,
        'framework': 'torch_tiny_sft',
        'parameter_count': sum(parameter.numel() for parameter in model.parameters()),
        'initial_loss': round(initial_loss, 6),
        'final_loss': round(final_loss, 6),
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
            'note': 'Tiny Torch SFT optimizes assistant-only next-token loss; helpfulness remains a separate proxy.',
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
