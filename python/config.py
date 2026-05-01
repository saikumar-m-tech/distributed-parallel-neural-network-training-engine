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


def get_config() -> FedSimConfig:
	if Path("/content").exists():
		return FedSimConfig(batch_size=512, epochs=20, limit_samples=20000)
	return FedSimConfig()
