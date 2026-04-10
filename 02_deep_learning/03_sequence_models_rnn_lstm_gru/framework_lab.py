from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'framework-manual'
METRICS_PATH = ARTIFACT_DIR / 'metrics.json'

SEQUENCE_TOKENS = ['A', 'B', 'C', 'X', 'Y', 'F']
TOKEN_TO_ID = {token: index for index, token in enumerate(SEQUENCE_TOKENS)}
EMBEDDINGS = torch.tensor(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.4, 0.2],
        [-1.0, 0.4, 0.2],
        [0.0, 0.0, 0.0],
    ],
    dtype=torch.float32,
)

DECODER_TOKENS = ['<pad>', '<bos>', '<eos>', 'red', 'blue', 'green']
DECODER_ID = {token: index for index, token in enumerate(DECODER_TOKENS)}
TRAINING_SEQUENCES = [['red', 'blue', 'green'], ['green', 'blue', 'red']]
DECODER_EMBED_DIM = 8
DECODER_HIDDEN_DIM = 10
DECODER_EPOCHS = 20
DECODER_LR = 0.05


class TinyGruDecoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(len(DECODER_TOKENS), DECODER_EMBED_DIM)
        self.gru = torch.nn.GRU(DECODER_EMBED_DIM, DECODER_HIDDEN_DIM, batch_first=True)
        self.output = torch.nn.Linear(DECODER_HIDDEN_DIM, len(DECODER_TOKENS))

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(input_ids)
        encoded, hidden = self.gru(embedded, hidden)
        return self.output(encoded), hidden

    def step(
        self,
        input_ids: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, hidden = self.forward(input_ids, hidden)
        return logits[:, -1:], hidden


def rounded(value: float) -> float:
    return round(float(value), 6)


def cosine_gap(left: torch.Tensor, right: torch.Tensor) -> float:
    return rounded(1.0 - float(F.cosine_similarity(left, right, dim=0).item()))


def set_recurrent_weights(rnn: torch.nn.RNN, lstm: torch.nn.LSTM, gru: torch.nn.GRU) -> None:
    with torch.no_grad():
        rnn.weight_ih_l0.copy_(
            torch.tensor([[0.9, -0.2, 0.3], [0.1, 0.8, -0.3], [0.2, 0.4, 0.7]], dtype=torch.float32)
        )
        rnn.weight_hh_l0.copy_(torch.diag(torch.tensor([0.45, 0.35, 0.25], dtype=torch.float32)))
        rnn.bias_ih_l0.zero_()
        rnn.bias_hh_l0.zero_()

        hidden_dim = lstm.hidden_size
        lstm.weight_ih_l0.zero_()
        lstm.weight_hh_l0.zero_()
        lstm.bias_ih_l0.zero_()
        lstm.bias_hh_l0.zero_()
        lstm.weight_ih_l0[0:hidden_dim].copy_(torch.eye(hidden_dim, dtype=torch.float32) * 1.1)
        lstm.weight_ih_l0[(2 * hidden_dim):(3 * hidden_dim)].copy_(torch.eye(hidden_dim, dtype=torch.float32))
        lstm.bias_ih_l0[0:hidden_dim].fill_(2.0)
        lstm.bias_ih_l0[hidden_dim:(2 * hidden_dim)].fill_(2.0)
        lstm.bias_ih_l0[(3 * hidden_dim):(4 * hidden_dim)].fill_(2.0)

        hidden_dim = gru.hidden_size
        gru.weight_ih_l0.zero_()
        gru.weight_hh_l0.zero_()
        gru.bias_ih_l0.zero_()
        gru.bias_hh_l0.zero_()
        gru.bias_ih_l0[0:hidden_dim].fill_(1.0)
        gru.bias_ih_l0[hidden_dim:(2 * hidden_dim)].fill_(2.0)
        gru.weight_ih_l0[(2 * hidden_dim):(3 * hidden_dim)].copy_(torch.eye(hidden_dim, dtype=torch.float32))
        gru.weight_hh_l0[(2 * hidden_dim):(3 * hidden_dim)].copy_(torch.eye(hidden_dim, dtype=torch.float32) * 0.25)


def build_decoder_tensors(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    sequences = [
        [DECODER_ID['<bos>'], *[DECODER_ID[token] for token in sequence], DECODER_ID['<eos>']]
        for sequence in TRAINING_SEQUENCES
    ]
    max_len = max(len(sequence) for sequence in sequences)
    decoder_inputs = []
    targets = []
    for sequence in sequences:
        padded = sequence + [DECODER_ID['<pad>']] * (max_len - len(sequence))
        decoder_inputs.append(padded[:-1])
        targets.append(padded[1:])
    return (
        torch.tensor(decoder_inputs, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
    )


def run() -> None:
    torch.manual_seed(5)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:  # pragma: no cover - defensive for older builds
        pass
    torch.set_num_threads(1)
    device = torch.device('cpu')

    embedding = torch.nn.Embedding.from_pretrained(EMBEDDINGS, freeze=True).to(device)
    rnn = torch.nn.RNN(input_size=3, hidden_size=3, batch_first=True, nonlinearity='tanh').to(device)
    lstm = torch.nn.LSTM(input_size=3, hidden_size=3, batch_first=True).to(device)
    gru = torch.nn.GRU(input_size=3, hidden_size=3, batch_first=True).to(device)
    set_recurrent_weights(rnn, lstm, gru)

    order_ids = torch.tensor(
        [
            [TOKEN_TO_ID['A'], TOKEN_TO_ID['B'], TOKEN_TO_ID['C']],
            [TOKEN_TO_ID['C'], TOKEN_TO_ID['B'], TOKEN_TO_ID['A']],
        ],
        dtype=torch.long,
        device=device,
    )
    long_context_ids = torch.tensor(
        [
            [TOKEN_TO_ID['X'], TOKEN_TO_ID['F'], TOKEN_TO_ID['F'], TOKEN_TO_ID['F'], TOKEN_TO_ID['F']],
            [TOKEN_TO_ID['Y'], TOKEN_TO_ID['F'], TOKEN_TO_ID['F'], TOKEN_TO_ID['F'], TOKEN_TO_ID['F']],
        ],
        dtype=torch.long,
        device=device,
    )
    order_inputs = embedding(order_ids)
    long_context_inputs = embedding(long_context_ids)

    with torch.no_grad():
        _, rnn_hidden = rnn(order_inputs)
        _, (lstm_hidden, lstm_cell) = lstm(order_inputs)
        _, gru_hidden = gru(order_inputs)

        _, rnn_long_hidden = rnn(long_context_inputs)
        _, (lstm_long_hidden, _) = lstm(long_context_inputs)
        _, gru_long_hidden = gru(long_context_inputs)

    decoder_inputs, targets = build_decoder_tensors(device)
    decoder = TinyGruDecoder().to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=DECODER_LR)
    loss_history: list[float] = []
    for _ in range(DECODER_EPOCHS):
        optimizer.zero_grad()
        logits, _ = decoder(decoder_inputs)
        loss = F.cross_entropy(
            logits.reshape(-1, len(DECODER_TOKENS)),
            targets.reshape(-1),
            ignore_index=DECODER_ID['<pad>'],
        )
        loss.backward()
        optimizer.step()
        loss_history.append(rounded(loss.item()))

    with torch.no_grad():
        decoder_logits, _ = decoder(decoder_inputs)
        teacher_forcing_loss = F.cross_entropy(
            decoder_logits.reshape(-1, len(DECODER_TOKENS)),
            targets.reshape(-1),
            ignore_index=DECODER_ID['<pad>'],
        )
        teacher_predictions = decoder_logits.argmax(dim=-1)

        token = torch.full((decoder_inputs.shape[0], 1), DECODER_ID['<bos>'], dtype=torch.long, device=device)
        hidden = None
        free_losses = []
        free_predictions = []
        for step in range(targets.shape[1]):
            step_logits, hidden = decoder.step(token, hidden)
            free_losses.append(
                F.cross_entropy(
                    step_logits.squeeze(1),
                    targets[:, step],
                    ignore_index=DECODER_ID['<pad>'],
                    reduction='none',
                )
            )
            predicted = step_logits.argmax(dim=-1)
            free_predictions.append(predicted.squeeze(1))
            token = predicted
        free_loss_tensor = torch.stack(free_losses, dim=1)
        non_pad_mask = (targets != DECODER_ID['<pad>']).float()
        free_running_loss = (free_loss_tensor * non_pad_mask).sum() / non_pad_mask.sum()
        free_predictions_tensor = torch.stack(free_predictions, dim=1)

    metrics = {
        'device': str(device),
        'order_input_shape': list(order_ids.shape),
        'long_context_shape': list(long_context_ids.shape),
        'embedding_dim': int(EMBEDDINGS.shape[1]),
        'hidden_dim': 3,
        'hidden_shapes': {
            'rnn': list(rnn_hidden.shape),
            'lstm_h': list(lstm_hidden.shape),
            'lstm_c': list(lstm_cell.shape),
            'gru': list(gru_hidden.shape),
        },
        'rnn_order_cosine_gap': cosine_gap(rnn_hidden[0, 0], rnn_hidden[0, 1]),
        'lstm_order_cosine_gap': cosine_gap(lstm_hidden[0, 0], lstm_hidden[0, 1]),
        'gru_order_cosine_gap': cosine_gap(gru_hidden[0, 0], gru_hidden[0, 1]),
        'rnn_long_range_signal': rounded(torch.norm(rnn_long_hidden[0, 0] - rnn_long_hidden[0, 1]).item()),
        'lstm_long_range_signal': rounded(torch.norm(lstm_long_hidden[0, 0] - lstm_long_hidden[0, 1]).item()),
        'gru_long_range_signal': rounded(torch.norm(gru_long_hidden[0, 0] - gru_long_hidden[0, 1]).item()),
        'decoder_logits_shape': list(decoder_logits.shape),
        'teacher_forcing_loss': rounded(teacher_forcing_loss.item()),
        'free_running_loss': rounded(free_running_loss.item()),
        'teacher_forcing_gap': rounded(free_running_loss.item() - teacher_forcing_loss.item()),
        'teacher_forcing_predictions': [
            [DECODER_TOKENS[token_id] for token_id in row]
            for row in teacher_predictions.cpu().tolist()
        ],
        'free_running_predictions': [
            [DECODER_TOKENS[token_id] for token_id in row]
            for row in free_predictions_tensor.cpu().tolist()
        ],
        'loss_history_head': loss_history[:5],
        'loss_history_tail': loss_history[-5:],
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
