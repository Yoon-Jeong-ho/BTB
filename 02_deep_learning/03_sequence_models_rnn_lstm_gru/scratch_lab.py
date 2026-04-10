from __future__ import annotations

import html
import json
import math
from pathlib import Path

import numpy as np

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'scratch-manual'
METRICS_PATH = ARTIFACT_DIR / 'metrics.json'
FIGURE_PATH = ARTIFACT_DIR / 'hidden_state_diagnostics.svg'

ORDER_PAIR = (['A', 'B', 'C'], ['C', 'B', 'A'])
LONG_CONTEXT_PAIR = (['X', 'F', 'F', 'F', 'F'], ['Y', 'F', 'F', 'F', 'F'])
TEACHER_TARGET = ['red', 'blue', 'green', '<eos>']

TOKEN_EMBEDDINGS = {
    'A': np.array([1.0, 0.0, 0.0], dtype=np.float64),
    'B': np.array([0.0, 1.0, 0.0], dtype=np.float64),
    'C': np.array([0.0, 0.0, 1.0], dtype=np.float64),
    'X': np.array([1.0, 0.4, 0.2], dtype=np.float64),
    'Y': np.array([-1.0, 0.4, 0.2], dtype=np.float64),
    'F': np.zeros(3, dtype=np.float64),
}

RNN_WX = np.array(
    [[0.9, -0.2, 0.3], [0.1, 0.8, -0.3], [0.2, 0.4, 0.7]],
    dtype=np.float64,
)
RNN_WH = np.diag(np.array([0.45, 0.35, 0.25], dtype=np.float64))

LSTM_WI = np.eye(3, dtype=np.float64) * 1.1
LSTM_UI = np.zeros((3, 3), dtype=np.float64)
LSTM_BI = np.ones(3, dtype=np.float64) * 2.0
LSTM_WF = np.zeros((3, 3), dtype=np.float64)
LSTM_UF = np.zeros((3, 3), dtype=np.float64)
LSTM_BF = np.ones(3, dtype=np.float64) * 2.0
LSTM_WG = np.eye(3, dtype=np.float64)
LSTM_UG = np.zeros((3, 3), dtype=np.float64)
LSTM_BG = np.zeros(3, dtype=np.float64)
LSTM_WO = np.zeros((3, 3), dtype=np.float64)
LSTM_UO = np.zeros((3, 3), dtype=np.float64)
LSTM_BO = np.ones(3, dtype=np.float64) * 2.0

GRU_WZ = np.zeros((3, 3), dtype=np.float64)
GRU_UZ = np.zeros((3, 3), dtype=np.float64)
GRU_BZ = np.ones(3, dtype=np.float64) * 2.0
GRU_WR = np.zeros((3, 3), dtype=np.float64)
GRU_UR = np.zeros((3, 3), dtype=np.float64)
GRU_BR = np.ones(3, dtype=np.float64)
GRU_WN = np.eye(3, dtype=np.float64)
GRU_UN = np.eye(3, dtype=np.float64) * 0.25
GRU_BN = np.zeros(3, dtype=np.float64)

TRANSITION_TOKENS = ['red', 'blue', 'green', '<eos>']
TRANSITIONS = {
    '<bos>': {'red': 2.0, 'blue': 2.25, 'green': -1.0, '<eos>': -3.0},
    'red': {'blue': 3.6, 'green': 0.8, 'red': -2.0, '<eos>': -1.5},
    'blue': {'green': 3.4, 'blue': 0.4, 'red': -1.5, '<eos>': -0.3},
    'green': {'<eos>': 3.8, 'green': 0.3, 'blue': -1.2, 'red': -1.2},
    '<eos>': {'<eos>': 3.0, 'red': -3.0, 'blue': -3.0, 'green': -3.0},
}


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def rounded(value: float) -> float:
    return round(float(value), 6)


def cosine_gap(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0.0:
        return 0.0
    cosine = float(np.dot(left, right) / denom)
    cosine = max(-1.0, min(1.0, cosine))
    return rounded(1.0 - cosine)


def rollout_rnn(sequence: list[str]) -> dict[str, np.ndarray]:
    hidden = np.zeros(3, dtype=np.float64)
    states = []
    for token in sequence:
        x_t = TOKEN_EMBEDDINGS[token]
        hidden = np.tanh(RNN_WX @ x_t + RNN_WH @ hidden)
        states.append(hidden.copy())
    return {'states': np.array(states, dtype=np.float64)}


def rollout_lstm(sequence: list[str]) -> dict[str, np.ndarray]:
    hidden = np.zeros(3, dtype=np.float64)
    cell = np.zeros(3, dtype=np.float64)
    states = []
    forget_gates = []
    for token in sequence:
        x_t = TOKEN_EMBEDDINGS[token]
        input_gate = sigmoid(LSTM_WI @ x_t + LSTM_UI @ hidden + LSTM_BI)
        forget_gate = sigmoid(LSTM_WF @ x_t + LSTM_UF @ hidden + LSTM_BF)
        candidate = np.tanh(LSTM_WG @ x_t + LSTM_UG @ hidden + LSTM_BG)
        output_gate = sigmoid(LSTM_WO @ x_t + LSTM_UO @ hidden + LSTM_BO)
        cell = (forget_gate * cell) + (input_gate * candidate)
        hidden = output_gate * np.tanh(cell)
        states.append(hidden.copy())
        forget_gates.append(forget_gate.copy())
    return {
        'states': np.array(states, dtype=np.float64),
        'forget_gates': np.array(forget_gates, dtype=np.float64),
    }


def rollout_gru(sequence: list[str]) -> dict[str, np.ndarray]:
    hidden = np.zeros(3, dtype=np.float64)
    states = []
    update_gates = []
    for token in sequence:
        x_t = TOKEN_EMBEDDINGS[token]
        update_gate = sigmoid(GRU_WZ @ x_t + GRU_UZ @ hidden + GRU_BZ)
        reset_gate = sigmoid(GRU_WR @ x_t + GRU_UR @ hidden + GRU_BR)
        candidate = np.tanh(GRU_WN @ x_t + GRU_UN @ (reset_gate * hidden) + GRU_BN)
        hidden = (update_gate * hidden) + ((1.0 - update_gate) * candidate)
        states.append(hidden.copy())
        update_gates.append(update_gate.copy())
    return {
        'states': np.array(states, dtype=np.float64),
        'update_gates': np.array(update_gates, dtype=np.float64),
    }


def transition_logits(previous_token: str) -> np.ndarray:
    return np.array(
        [TRANSITIONS.get(previous_token, {}).get(token, -4.0) for token in TRANSITION_TOKENS],
        dtype=np.float64,
    )


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def evaluate_teacher_forcing_gap() -> dict[str, object]:
    teacher_forcing_losses: list[float] = []
    teacher_predictions: list[str] = []
    previous = '<bos>'
    for target in TEACHER_TARGET:
        probabilities = softmax(transition_logits(previous))
        target_index = TRANSITION_TOKENS.index(target)
        teacher_forcing_losses.append(-math.log(float(probabilities[target_index])))
        teacher_predictions.append(TRANSITION_TOKENS[int(np.argmax(probabilities))])
        previous = target

    free_running_losses: list[float] = []
    free_predictions: list[str] = []
    previous = '<bos>'
    for target in TEACHER_TARGET:
        probabilities = softmax(transition_logits(previous))
        target_index = TRANSITION_TOKENS.index(target)
        free_running_losses.append(-math.log(float(probabilities[target_index])))
        prediction = TRANSITION_TOKENS[int(np.argmax(probabilities))]
        free_predictions.append(prediction)
        previous = prediction

    teacher_loss = float(np.mean(teacher_forcing_losses))
    free_loss = float(np.mean(free_running_losses))
    return {
        'target_tokens': TEACHER_TARGET,
        'teacher_forcing_predictions': teacher_predictions,
        'free_running_predictions': free_predictions,
        'teacher_forcing_loss': rounded(teacher_loss),
        'free_running_loss': rounded(free_loss),
        'teacher_forcing_gap': rounded(free_loss - teacher_loss),
    }


def _polyline(points: list[tuple[float, float]], color: str) -> str:
    point_text = ' '.join(f'{x:.2f},{y:.2f}' for x, y in points)
    return (
        f'<polyline fill="none" stroke="{color}" stroke-width="3" '
        f'points="{point_text}" />'
    )


def save_svg(order_a: np.ndarray, order_b: np.ndarray, retention: dict[str, float], losses: dict[str, float]) -> None:
    width, height = 820, 480
    line_left, line_right = 60, 320
    line_top, line_bottom = 100, 290
    bar_left, bar_right = 420, 770
    bar_top = 110
    loss_top, loss_bottom = 340, 430
    steps = len(order_a)
    x_positions = np.linspace(line_left, line_right, steps)
    all_values = np.concatenate([order_a[:, 0], order_b[:, 0]])
    y_min = float(np.min(all_values) - 0.1)
    y_max = float(np.max(all_values) + 0.1)

    def map_line_y(value: float) -> float:
        return line_bottom - ((value - y_min) / (y_max - y_min)) * (line_bottom - line_top)

    order_a_points = [(float(x_positions[index]), map_line_y(float(value))) for index, value in enumerate(order_a[:, 0])]
    order_b_points = [(float(x_positions[index]), map_line_y(float(value))) for index, value in enumerate(order_b[:, 0])]

    max_retention = max(retention.values()) or 1.0
    retention_rows = []
    retention_labels = []
    for index, (label, value) in enumerate(retention.items()):
        y = bar_top + (index * 52)
        bar_width = (float(value) / max_retention) * (bar_right - bar_left - 120)
        retention_rows.append(
            f'<rect x="{bar_left + 110}" y="{y - 14}" width="{bar_width:.2f}" height="22" fill="#74c0fc" rx="4" />'
        )
        retention_labels.append(
            f'<text x="{bar_left}" y="{y}" font-size="14" font-family="Arial, sans-serif" fill="#222">{html.escape(label)}</text>'
            f'<text x="{bar_left + 120 + bar_width:.2f}" y="{y}" font-size="13" font-family="Arial, sans-serif" fill="#495057">{value:.3f}</text>'
        )

    loss_max = max(losses.values()) or 1.0
    loss_rows = []
    for index, (label, value) in enumerate(losses.items()):
        x = bar_left + (index * 150)
        bar_height = (float(value) / loss_max) * (loss_bottom - loss_top - 20)
        y = loss_bottom - bar_height
        color = '#ff922b' if 'free' in label else '#51cf66'
        loss_rows.append(
            f'<rect x="{x}" y="{y:.2f}" width="70" height="{bar_height:.2f}" fill="{color}" rx="4" />'
            f'<text x="{x}" y="{loss_bottom + 18}" font-size="14" font-family="Arial, sans-serif" fill="#222">{html.escape(label)}</text>'
            f'<text x="{x}" y="{y - 8:.2f}" font-size="13" font-family="Arial, sans-serif" fill="#495057">{value:.3f}</text>'
        )

    step_labels = []
    for index, x in enumerate(x_positions):
        step_labels.append(
            f'<text x="{x:.2f}" y="{line_bottom + 24}" text-anchor="middle" font-size="12" font-family="Arial, sans-serif" fill="#495057">t{index + 1}</text>'
        )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="40" y="38" font-size="24" font-family="Arial, sans-serif" fill="#111">Hidden state diagnostics</text>
  <text x="40" y="62" font-size="14" font-family="Arial, sans-serif" fill="#495057">order sensitivity, long-range retention, teacher forcing gap</text>

  <text x="{line_left}" y="88" font-size="16" font-family="Arial, sans-serif" fill="#111">Vanilla RNN hidden dim-0 trajectory</text>
  <line x1="{line_left}" y1="{line_bottom}" x2="{line_right}" y2="{line_bottom}" stroke="#222" stroke-width="2" />
  <line x1="{line_left}" y1="{line_top}" x2="{line_left}" y2="{line_bottom}" stroke="#222" stroke-width="2" />
  {_polyline(order_a_points, '#2b8a3e')}
  {_polyline(order_b_points, '#c92a2a')}
  <text x="{line_right - 80}" y="{line_top + 16}" font-size="13" font-family="Arial, sans-serif" fill="#2b8a3e">A→B→C</text>
  <text x="{line_right - 80}" y="{line_top + 36}" font-size="13" font-family="Arial, sans-serif" fill="#c92a2a">C→B→A</text>
  {''.join(step_labels)}

  <text x="{bar_left}" y="88" font-size="16" font-family="Arial, sans-serif" fill="#111">Long-range signal retention</text>
  {''.join(retention_rows)}
  {''.join(retention_labels)}

  <text x="{bar_left}" y="324" font-size="16" font-family="Arial, sans-serif" fill="#111">Decoder loss: teacher forcing vs free running</text>
  {''.join(loss_rows)}
</svg>
'''
    FIGURE_PATH.write_text(svg, encoding='utf-8')


def run() -> None:
    rnn_order_a = rollout_rnn(ORDER_PAIR[0])
    rnn_order_b = rollout_rnn(ORDER_PAIR[1])
    lstm_order_a = rollout_lstm(ORDER_PAIR[0])
    lstm_order_b = rollout_lstm(ORDER_PAIR[1])
    gru_order_a = rollout_gru(ORDER_PAIR[0])
    gru_order_b = rollout_gru(ORDER_PAIR[1])

    rnn_long_a = rollout_rnn(LONG_CONTEXT_PAIR[0])
    rnn_long_b = rollout_rnn(LONG_CONTEXT_PAIR[1])
    lstm_long_a = rollout_lstm(LONG_CONTEXT_PAIR[0])
    lstm_long_b = rollout_lstm(LONG_CONTEXT_PAIR[1])
    gru_long_a = rollout_gru(LONG_CONTEXT_PAIR[0])
    gru_long_b = rollout_gru(LONG_CONTEXT_PAIR[1])

    teacher = evaluate_teacher_forcing_gap()
    retention = {
        'vanilla RNN': rounded(np.linalg.norm(rnn_long_a['states'][-1] - rnn_long_b['states'][-1])),
        'LSTM': rounded(np.linalg.norm(lstm_long_a['states'][-1] - lstm_long_b['states'][-1])),
        'GRU': rounded(np.linalg.norm(gru_long_a['states'][-1] - gru_long_b['states'][-1])),
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    save_svg(
        rnn_order_a['states'],
        rnn_order_b['states'],
        retention,
        {
            'teacher forcing': teacher['teacher_forcing_loss'],
            'free running': teacher['free_running_loss'],
        },
    )

    metrics = {
        'sequence_pairs': [
            {'left': ORDER_PAIR[0], 'right': ORDER_PAIR[1]},
            {'left': LONG_CONTEXT_PAIR[0], 'right': LONG_CONTEXT_PAIR[1]},
        ],
        'rnn_order_cosine_gap': cosine_gap(rnn_order_a['states'][-1], rnn_order_b['states'][-1]),
        'lstm_order_cosine_gap': cosine_gap(lstm_order_a['states'][-1], lstm_order_b['states'][-1]),
        'gru_order_cosine_gap': cosine_gap(gru_order_a['states'][-1], gru_order_b['states'][-1]),
        'rnn_long_range_signal': retention['vanilla RNN'],
        'lstm_long_range_signal': retention['LSTM'],
        'gru_long_range_signal': retention['GRU'],
        'lstm_forget_gate_mean': rounded(float(np.mean(lstm_long_a['forget_gates']))),
        'gru_update_gate_mean': rounded(float(np.mean(gru_long_a['update_gates']))),
        'teacher_target_tokens': teacher['target_tokens'],
        'teacher_forcing_predictions': teacher['teacher_forcing_predictions'],
        'free_running_predictions': teacher['free_running_predictions'],
        'teacher_forcing_loss': teacher['teacher_forcing_loss'],
        'free_running_loss': teacher['free_running_loss'],
        'teacher_forcing_gap': teacher['teacher_forcing_gap'],
        'figure_path': str(FIGURE_PATH.relative_to(UNIT_ROOT)),
    }

    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
