from __future__ import annotations

import json
from html import escape
from pathlib import Path


UNIT = "vision_language_action_grounding"
ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts" / "scratch-manual"
ACTIONS = ["pick_red_block", "push_blue_cube", "stop_before_hazard", "place_green_goal"]
SCENARIOS = [
    {
        "scene": "red block on table",
        "instruction": "빨간 블록을 집어라",
        "target_action": "pick_red_block",
        "safe_to_execute": True,
    },
    {
        "scene": "blue cube near target lane",
        "instruction": "파란 큐브를 목표선 쪽으로 밀어라",
        "target_action": "push_blue_cube",
        "safe_to_execute": True,
    },
    {
        "scene": "robot arm faces a hazard zone",
        "instruction": "앞으로 이동하라, 단 위험하면 멈춰라",
        "target_action": "stop_before_hazard",
        "safe_to_execute": False,
    },
    {
        "scene": "green marker beside goal bin",
        "instruction": "초록 마커를 목표 바구니에 놓아라",
        "target_action": "place_green_goal",
        "safe_to_execute": True,
    },
]


def policy_scores() -> list[list[float]]:
    return [
        [3.4, 0.4, -0.7, 0.2],
        [0.2, 3.2, -0.5, 0.7],
        [-0.8, -0.6, 3.6, -0.4],
        [0.3, 0.6, -0.6, 3.3],
    ]


def argmax(values: list[float]) -> int:
    return max(range(len(values)), key=lambda index: values[index])


def write_svg(scores: list[list[float]], path: Path) -> None:
    cell = 78
    left = 170
    top = 58
    width = left + cell * len(ACTIONS) + 30
    height = top + cell * len(SCENARIOS) + 70
    max_score = max(max(row) for row in scores)
    min_score = min(min(row) for row in scores)

    def color(value: float) -> str:
        ratio = (value - min_score) / (max_score - min_score)
        red = int(244 - ratio * 116)
        green = int(236 - ratio * 136)
        blue = int(220 - ratio * 165)
        return f"rgb({red},{green},{blue})"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffaf2"/>',
        '<text x="24" y="32" font-size="20" font-weight="700" fill="#201a17">VLA action policy matrix</text>',
    ]
    for col, action in enumerate(ACTIONS):
        x = left + col * cell + cell / 2
        parts.append(
            f'<text x="{x}" y="50" text-anchor="middle" font-size="11" fill="#5b5049">{escape(action)}</text>'
        )
    for row, scenario in enumerate(SCENARIOS):
        y = top + row * cell + cell / 2
        parts.append(
            f'<text x="18" y="{y}" font-size="12" fill="#5b5049">{escape(scenario["target_action"])}</text>'
        )
        for col, value in enumerate(scores[row]):
            x = left + col * cell
            y0 = top + row * cell
            parts.append(f'<rect x="{x}" y="{y0}" width="{cell - 4}" height="{cell - 4}" rx="10" fill="{color(value)}"/>')
            parts.append(
                f'<text x="{x + cell / 2}" y="{y0 + cell / 2 + 5}" text-anchor="middle" font-size="16" font-weight="700" fill="#201a17">{value:.1f}</text>'
            )
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    scores = policy_scores()
    predicted_indices = [argmax(row) for row in scores]
    target_indices = [ACTIONS.index(scenario["target_action"]) for scenario in SCENARIOS]
    predicted_actions = [ACTIONS[index] for index in predicted_indices]
    safety_predictions = [action != "stop_before_hazard" for action in predicted_actions]
    safety_targets = [scenario["safe_to_execute"] for scenario in SCENARIOS]

    figure_path = ARTIFACT_DIR / "action_policy_matrix.svg"
    write_svg(scores, figure_path)

    payload = {
        "unit": UNIT,
        "scenario_count": len(SCENARIOS),
        "actions": ACTIONS,
        "policy_matrix_shape": [len(scores), len(scores[0])],
        "predicted_actions": predicted_actions,
        "target_actions": [scenario["target_action"] for scenario in SCENARIOS],
        "action_accuracy": sum(p == t for p, t in zip(predicted_indices, target_indices)) / len(SCENARIOS),
        "safety_predictions": safety_predictions,
        "safety_targets": safety_targets,
        "safety_gate_accuracy": sum(p == t for p, t in zip(safety_predictions, safety_targets)) / len(SCENARIOS),
        "figure_path": "artifacts/scratch-manual/action_policy_matrix.svg",
    }
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
