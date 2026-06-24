from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


UNIT = "vision_language_action_grounding"
ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts" / "framework-manual"


class TinyVLAPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(6, 8)
        self.action_head = nn.Linear(8, 4)
        self.safety_head = nn.Linear(8, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.tanh(self.encoder(features))
        return self.action_head(hidden), self.safety_head(hidden).squeeze(-1)


def toy_batch(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # red + pick instruction
            [0.0, 1.0, 0.0, 0.0, 0.6, 0.3],  # blue + push instruction
            [0.0, 0.0, 1.0, 1.0, 0.2, 1.0],  # hazard + stop condition
            [0.0, 0.0, 0.0, 1.0, 0.8, 0.0],  # green goal placement
        ],
        dtype=torch.float32,
        device=device,
    )
    action_targets = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=device)
    safety_targets = torch.tensor([1.0, 1.0, 0.0, 1.0], dtype=torch.float32, device=device)
    return features, action_targets, safety_targets


def train_policy(device: torch.device) -> dict[str, object]:
    torch.manual_seed(7)
    features, action_targets, safety_targets = toy_batch(device)
    model = TinyVLAPolicy().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.08, weight_decay=0.0)
    loss_history: list[float] = []

    for _ in range(100):
        optimizer.zero_grad()
        logits, safety_logits = model(features)
        action_loss = F.cross_entropy(logits, action_targets)
        safety_loss = F.binary_cross_entropy_with_logits(safety_logits, safety_targets)
        loss = action_loss + 0.6 * safety_loss
        loss.backward()
        optimizer.step()
        loss_history.append(round(float(loss.detach().cpu()), 6))

    with torch.no_grad():
        logits, safety_logits = model(features)
        predictions = logits.argmax(dim=-1)
        safety_predictions = (torch.sigmoid(safety_logits) >= 0.5).float()
        action_accuracy = float((predictions == action_targets).float().mean().cpu())
        safety_gate_accuracy = float((safety_predictions == safety_targets).float().mean().cpu())

    return {
        "unit": UNIT,
        "device": device.type,
        "feature_shape": list(features.shape),
        "logits_shape": list(logits.shape),
        "action_predictions": predictions.cpu().tolist(),
        "action_targets": action_targets.cpu().tolist(),
        "action_accuracy": round(action_accuracy, 6),
        "safety_predictions": [bool(x) for x in safety_predictions.cpu().tolist()],
        "safety_targets": [bool(x) for x in safety_targets.cpu().tolist()],
        "safety_gate_accuracy": round(safety_gate_accuracy, 6),
        "loss_history_head": loss_history[:5],
        "loss_history_tail": loss_history[-5:],
    }


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    payload = train_policy(device)
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
