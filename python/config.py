from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FedSimConfig:
	data_dir: str = "data"
	batch_size: int = 128
	epochs: int = 10
	learning_rate: float = 0.01
	limit_samples: int | None = None
	status_accuracy_samples: int = 2000
	status_accuracy_every: int = 10
	ready_steps: int = 50
	queue_max_batches: int = 64


def get_config() -> FedSimConfig:
	if Path("/content").exists():
		return FedSimConfig(batch_size=512, epochs=20, limit_samples=20000)
	return FedSimConfig()
