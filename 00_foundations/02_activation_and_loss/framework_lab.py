from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'framework-manual'
METRICS_PATH = ARTIFACT_DIR / 'metrics.json'


def activation_rows(inputs: torch.Tensor, relu: torch.Tensor, sigmoid: torch.Tensor, tanh: torch.Tensor) -> list[dict[str, float]]:
    rows = []
    for index, value in enumerate(inputs.flatten().tolist()):
        rows.append(
            {
                'input': round(float(value), 6),
                'relu': round(float(relu.flatten()[index]), 6),
                'sigmoid': round(float(sigmoid.flatten()[index]), 6),
                'tanh': round(float(tanh.flatten()[index]), 6),
            }
        )
    return rows


def numeric_stability_demo() -> dict[str, float | str]:
    extreme_logits = torch.tensor([1000.0], dtype=torch.float32)
    wrong_target = torch.tensor([0.0], dtype=torch.float32)
    stable_loss = F.binary_cross_entropy_with_logits(extreme_logits, wrong_target)
    saturated_probability = torch.sigmoid(extreme_logits)
    naive_loss = F.binary_cross_entropy(saturated_probability, wrong_target)
    return {
        'extreme_logit': round(float(extreme_logits.item()), 6),
        'sigmoid_probability': round(float(saturated_probability.item()), 6),
        'stable_bce_with_logits': round(float(stable_loss.item()), 6),
        'naive_bce_after_sigmoid': str(float(naive_loss.item())),
        'why': 'sigmoid를 먼저 적용하면 극단 logit이 0 또는 1로 포화되어 확률 공간에서 정보가 사라진다. BCEWithLogitsLoss는 sigmoid와 log를 합친 안정식으로 계산해 유한한 loss를 유지한다.',
    }


def run() -> None:
    torch.manual_seed(7)

    # 같은 입력값을 세 activation에 모두 통과시켜, 각 함수가 숫자를 어떻게 바꾸는지 비교한다.
    activation_inputs = torch.tensor(
        [[-2.0, -0.5, 0.0, 0.5, 2.0], [1.5, -1.0, 0.25, -0.25, 0.75]],
        dtype=torch.float32,
    )
    # ReLU는 음수를 0으로 잘라 sparse한 활성값을 만든다.
    relu_values = torch.relu(activation_inputs)
    # sigmoid는 모든 값을 0~1 사이 확률처럼 읽을 수 있게 누른다.
    sigmoid_values = torch.sigmoid(activation_inputs)
    # tanh는 값을 -1~1 사이로 누르되 0을 중심으로 양수/음수 방향을 보존한다.
    tanh_values = torch.tanh(activation_inputs)

    # multi-class 분류에서는 raw score인 logits를 CrossEntropyLoss에 바로 넣는다.
    # softmax는 사람이 확률처럼 읽기 위한 관측값이고, loss 계산에는 logits를 쓰는 것이 수치적으로 안정적이다.
    class_logits = torch.tensor(
        [[2.2, 0.3, -1.4], [0.1, 1.7, -0.5]],
        dtype=torch.float32,
    )
    class_targets = torch.tensor([0, 1], dtype=torch.long)
    class_probabilities = torch.softmax(class_logits, dim=-1)
    cross_entropy_loss = F.cross_entropy(class_logits, class_targets)

    # binary 분류에서도 BCEWithLogitsLoss는 sigmoid를 따로 적용하지 않은 logits를 받는다.
    # 아래 binary_probabilities는 loss 입력이 아니라, 결과를 사람이 해석하기 위한 확률 관측값이다.
    binary_logits = torch.tensor([1.25, -0.75, 0.2, -1.5], dtype=torch.float32)
    binary_targets = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    binary_probabilities = torch.sigmoid(binary_logits)
    binary_cross_entropy_loss = F.binary_cross_entropy_with_logits(binary_logits, binary_targets)

    metrics = {
        'activation_reading_guide': '같은 input에 대해 ReLU는 음수를 0으로 자르고, sigmoid는 0~1 확률 범위로 누르며, tanh는 -1~1 centered 범위로 누릅니다.',
        'activation_input_shape': list(activation_inputs.shape),
        'activation_rows': activation_rows(activation_inputs, relu_values, sigmoid_values, tanh_values),
        'relu_zero_fraction': round(float((relu_values == 0).float().mean()), 6),
        'relu_first_row': [round(float(value), 6) for value in relu_values[0]],
        'sigmoid_first_row': [round(float(value), 6) for value in sigmoid_values[0]],
        'tanh_first_row': [round(float(value), 6) for value in tanh_values[0]],
        'activation_summary': {
            'relu': '음수 입력은 0이 되어 다음 층으로 신호가 가지 않는다.',
            'sigmoid': '큰 음수는 0에 가깝고 큰 양수는 1에 가까워 binary probability처럼 읽힌다.',
            'tanh': 'sigmoid와 비슷하게 포화되지만 출력 중심이 0이라 음수 방향 정보가 남는다.',
        },
        'class_logits_shape': list(class_logits.shape),
        'row_probability_sums': [round(float(value), 6) for value in class_probabilities.sum(dim=-1)],
        'target_class_probabilities': [
            round(float(class_probabilities[index, target]), 6)
            for index, target in enumerate(class_targets.tolist())
        ],
        'cross_entropy_loss': round(float(cross_entropy_loss), 6),
        'loss_reading_guide': '확률을 눈으로 확인할 때는 sigmoid/softmax 값을 보지만, PyTorch loss 함수에는 보통 logits를 직접 넣는다.',
        'numeric_stability_demo': numeric_stability_demo(),
        'binary_probabilities': [round(float(value), 6) for value in binary_probabilities],
        'binary_cross_entropy_loss': round(float(binary_cross_entropy_loss), 6),
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
