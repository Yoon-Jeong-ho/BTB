from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'framework-manual'
METRICS_PATH = ARTIFACT_DIR / 'metrics.json'
TOKENS = ['<pad>', '<bos>', '<eos>', 'a', 'cat', 'dog', 'kite', 'bowl', 'on', 'over', 'of', 'mat', 'beach', 'soup']
TOKEN_TO_ID = {token: index for index, token in enumerate(TOKENS)}
CONTENT_TOKENS = {'cat', 'dog', 'kite', 'bowl', 'mat', 'beach', 'soup'}
EPOCHS = 60
LEARNING_RATE = 0.05
EMBED_DIM = 16
HIDDEN_DIM = 20


def build_toy_dataset() -> tuple[torch.Tensor, list[list[str]], list[str]]:
    image_inputs = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    captions = [
        ['a', 'cat', 'on', 'mat'],
        ['a', 'kite', 'over', 'beach'],
        ['a', 'bowl', 'of', 'soup'],
        ['a', 'dog', 'on', 'beach'],
    ]
    image_labels = [
        '실내 고양이 매트',
        '해변 위 연',
        '수프가 담긴 그릇',
        '해변을 걷는 강아지',
    ]
    return image_inputs, captions, image_labels


def build_training_tensors(captions: list[list[str]]) -> tuple[torch.Tensor, torch.Tensor]:
    sequences = [[TOKEN_TO_ID['<bos>'], *[TOKEN_TO_ID[token] for token in caption], TOKEN_TO_ID['<eos>']] for caption in captions]
    max_len = max(len(sequence) for sequence in sequences)
    decoder_inputs = []
    targets = []
    for sequence in sequences:
        padded = sequence + [TOKEN_TO_ID['<pad>']] * (max_len - len(sequence))
        decoder_inputs.append(padded[:-1])
        targets.append(padded[1:])
    return torch.tensor(decoder_inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def _validate_caption_batch(image_inputs: torch.Tensor, decoder_inputs: torch.Tensor) -> None:
    if image_inputs.ndim != 2 or decoder_inputs.ndim != 2:
        raise ValueError(
            'framework image captioning example expects 2D tensors shaped like '
            '(batch, feature_dim) for image_inputs and (batch, steps) for decoder_inputs.'
        )
    if image_inputs.shape[0] != decoder_inputs.shape[0]:
        raise ValueError(
            'image/token batch size must match for this image captioning toy setup: '
            f'got image batch {image_inputs.shape[0]} and decoder batch {decoder_inputs.shape[0]}.'
        )


class TinyCaptionDecoder(torch.nn.Module):
    def __init__(self, image_dim: int, vocab_size: int, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.image_projection = torch.nn.Linear(image_dim, hidden_dim)
        self.token_embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.decoder = torch.nn.GRU(embed_dim + hidden_dim, hidden_dim, batch_first=True)
        self.output_projection = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, image_inputs: torch.Tensor, decoder_inputs: torch.Tensor) -> torch.Tensor:
        _validate_caption_batch(image_inputs, decoder_inputs)
        context = self.image_projection(image_inputs)
        repeated_context = context.unsqueeze(1).expand(-1, decoder_inputs.shape[1], -1)
        token_embeddings = self.token_embedding(decoder_inputs)
        decoder_features = torch.cat([token_embeddings, repeated_context], dim=-1)
        hidden0 = context.unsqueeze(0)
        decoder_outputs, _ = self.decoder(decoder_features, hidden0)
        return self.output_projection(decoder_outputs)

    def greedy_decode(self, image_inputs: torch.Tensor, max_steps: int) -> torch.Tensor:
        batch_size = image_inputs.shape[0]
        hidden = self.image_projection(image_inputs).unsqueeze(0)
        token = torch.full(
            (batch_size, 1),
            TOKEN_TO_ID['<bos>'],
            dtype=torch.long,
            device=image_inputs.device,
        )
        decoded_steps = []
        for _ in range(max_steps):
            token_embedding = self.token_embedding(token)
            context = hidden.transpose(0, 1)
            decoder_features = torch.cat([token_embedding, context], dim=-1)
            decoder_output, hidden = self.decoder(decoder_features, hidden)
            next_token = self.output_projection(decoder_output[:, -1]).argmax(dim=-1, keepdim=True)
            decoded_steps.append(next_token)
            token = next_token
        return torch.cat(decoded_steps, dim=1)


def compute_caption_logits(
    image_inputs: torch.Tensor,
    decoder_inputs: torch.Tensor,
    model: TinyCaptionDecoder | None = None,
) -> torch.Tensor:
    _validate_caption_batch(image_inputs, decoder_inputs)
    if model is None:
        model = TinyCaptionDecoder(
            image_dim=image_inputs.shape[1],
            vocab_size=len(TOKENS),
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
        )
    return model(image_inputs, decoder_inputs)


def decode_token_ids(token_ids: list[int]) -> list[str]:
    decoded: list[str] = []
    for token_id in token_ids:
        token = TOKENS[token_id]
        if token == '<eos>':
            break
        if token != '<pad>':
            decoded.append(token)
    return decoded


def _content_overlap(reference: list[str], generated: list[str]) -> tuple[int, int]:
    reference_content = [token for token in reference if token in CONTENT_TOKENS]
    generated_content = [token for token in generated if token in CONTENT_TOKENS]
    overlap = sum(token in reference_content for token in generated_content)
    hallucinated = sum(token not in reference_content for token in generated_content)
    return overlap, hallucinated


def _safe_precision(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator / denominator), 6)


def run() -> None:
    torch.manual_seed(7)
    device = torch.device('cpu')

    image_inputs, captions, image_labels = build_toy_dataset()
    image_inputs = image_inputs.to(device)
    decoder_inputs, targets = build_training_tensors(captions)
    decoder_inputs = decoder_inputs.to(device)
    targets = targets.to(device)

    model = TinyCaptionDecoder(
        image_dim=image_inputs.shape[1],
        vocab_size=len(TOKENS),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    loss_history: list[float] = []
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        logits = model(image_inputs, decoder_inputs)
        loss = F.cross_entropy(
            logits.reshape(-1, len(TOKENS)),
            targets.reshape(-1),
            ignore_index=TOKEN_TO_ID['<pad>'],
        )
        loss.backward()
        optimizer.step()
        loss_history.append(round(float(loss.item()), 6))

    with torch.no_grad():
        logits = model(image_inputs, decoder_inputs)
        predictions = logits.argmax(dim=-1)
        non_pad_mask = targets != TOKEN_TO_ID['<pad>']
        token_accuracy = float((predictions[non_pad_mask] == targets[non_pad_mask]).float().mean().item())
        decoded = model.greedy_decode(image_inputs, max_steps=targets.shape[1])

    generated_rows = []
    exact_matches = []
    overlap_total = 0
    generated_content_total = 0
    hallucinated_total = 0
    caption_lengths = []

    for image_label, reference_tokens, generated_ids in zip(image_labels, captions, decoded.tolist()):
        generated_tokens = decode_token_ids(generated_ids)
        overlap, hallucinated = _content_overlap(reference_tokens, generated_tokens)
        generated_content = [token for token in generated_tokens if token in CONTENT_TOKENS]
        overlap_total += overlap
        generated_content_total += len(generated_content)
        hallucinated_total += hallucinated
        caption_lengths.append(len(generated_tokens))
        is_exact_match = generated_tokens == reference_tokens
        exact_matches.append(is_exact_match)
        generated_rows.append(
            {
                'image_label': image_label,
                'reference_caption': ' '.join(reference_tokens),
                'generated_caption': ' '.join(generated_tokens),
                'is_exact_match': is_exact_match,
            }
        )

    metrics = {
        'device': str(device),
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
        'vocab_size': len(TOKENS),
        'image_input_shape': list(image_inputs.shape),
        'decoder_input_shape': list(decoder_inputs.shape),
        'target_shape': list(targets.shape),
        'embedding_dim': EMBED_DIM,
        'hidden_dim': HIDDEN_DIM,
        'loss_history_head': loss_history[:5],
        'loss_history_tail': loss_history[-5:],
        'token_accuracy': round(token_accuracy, 6),
        'exact_match_rate': round(float(sum(exact_matches) / len(exact_matches)), 6),
        'corpus_unigram_precision': _safe_precision(overlap_total, generated_content_total),
        'mean_caption_length': round(float(sum(caption_lengths) / len(caption_lengths)), 6),
        'hallucinated_content_tokens_total': int(hallucinated_total),
        'generated_rows': generated_rows,
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
