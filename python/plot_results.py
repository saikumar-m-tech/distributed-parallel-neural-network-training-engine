from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _ensure_output_path(output: str | Path) -> Path:
	path = Path(output)
	path.parent.mkdir(parents=True, exist_ok=True)
	return path


def plot_convergence(csv_path, output="plots/convergence.png"):
	data = pd.read_csv(csv_path)

	fig, ax_left = plt.subplots(figsize=(8, 5))
	ax_right = ax_left.twinx()

	ax_left.plot(data["epoch"], data["loss"], color="tab:blue", label="Loss")
	ax_right.plot(data["epoch"], data["accuracy"], color="tab:orange", label="Accuracy")

	ax_left.set_xlabel("Epoch")
	ax_left.set_ylabel("Loss")
	ax_right.set_ylabel("Accuracy")
	ax_left.set_title("Training Convergence")

	left_handles, left_labels = ax_left.get_legend_handles_labels()
	right_handles, right_labels = ax_right.get_legend_handles_labels()
	ax_left.legend(left_handles + right_handles, left_labels + right_labels, loc="best")

	fig.tight_layout()
	fig.savefig(_ensure_output_path(output))
	plt.close(fig)


def plot_scaling_efficiency(results_dict, output="plots/scaling.png"):
	workers = sorted(results_dict)
	times = [results_dict[worker]["time_per_epoch"] for worker in workers]

	baseline_workers = 1 if 1 in results_dict else workers[0]
	baseline_time = results_dict[baseline_workers]["time_per_epoch"]
	ideal_times = [baseline_time * baseline_workers / worker for worker in workers]

	actual_speedup = [baseline_time / time for time in times]
	ideal_speedup = [worker / baseline_workers for worker in workers]
	efficiency = [100.0 * actual / ideal for actual, ideal in zip(actual_speedup, ideal_speedup)]

	fig, ax_left = plt.subplots(figsize=(8, 5))
	ax_left.bar(workers, times, color="tab:blue", alpha=0.7, label="Actual time/epoch")
	ax_left.plot(workers, ideal_times, color="tab:orange", marker="o", label="Ideal linear time")

	ax_left.set_xlabel("Workers")
	ax_left.set_ylabel("Time per epoch (s)")
	ax_left.set_title("Scaling Efficiency")
	ax_left.set_xticks(workers)

	ax_right = ax_left.twinx()
	ax_right.plot(
		workers,
		efficiency,
		color="tab:green",
		marker="s",
		linestyle="--",
		label="Parallel efficiency (%)",
	)
	ax_right.set_ylabel("Parallel efficiency (%)")

	left_handles, left_labels = ax_left.get_legend_handles_labels()
	right_handles, right_labels = ax_right.get_legend_handles_labels()
	ax_left.legend(left_handles + right_handles, left_labels + right_labels, loc="best")

	fig.tight_layout()
	fig.savefig(_ensure_output_path(output))
	plt.close(fig)


def plot_communication_breakdown(results_dict, output="plots/comm_breakdown.png"):
	workers = sorted(results_dict)
	compute = [results_dict[worker]["compute_ms"] for worker in workers]
	sync = [results_dict[worker]["sync_ms"] for worker in workers]

	fig, ax = plt.subplots(figsize=(8, 5))
	ax.bar(workers, compute, color="tab:blue", label="Compute")
	ax.bar(workers, sync, bottom=compute, color="tab:orange", label="Communication")

	ax.set_xlabel("Workers")
	ax.set_ylabel("Time per epoch (ms)")
	ax.set_title("Communication Breakdown")
	ax.set_xticks(workers)
	ax.legend(loc="best")

	fig.tight_layout()
	fig.savefig(_ensure_output_path(output))
	plt.close(fig)


def plot_accuracy_parity(parallelnet_csv, pytorch_csv, output="plots/parity.png"):
	parallelnet = pd.read_csv(parallelnet_csv)
	pytorch = pd.read_csv(pytorch_csv)

	fig, ax = plt.subplots(figsize=(8, 5))
	ax.plot(parallelnet["epoch"], parallelnet["loss"], label="ParallelNet loss")
	ax.plot(pytorch["epoch"], pytorch["loss"], label="PyTorch loss")

	ax.set_xlabel("Epoch")
	ax.set_ylabel("Loss")
	ax.set_title("Loss Parity")
	ax.legend(loc="best")

	fig.tight_layout()
	fig.savefig(_ensure_output_path(output))
	plt.close(fig)


if __name__ == "__main__":
	mock_dir = Path("plots/mock_data")
	mock_dir.mkdir(parents=True, exist_ok=True)

	epochs = list(range(1, 11))
	loss = [1.2 / epoch + 0.05 for epoch in epochs]
	accuracy = [min(0.45 + 0.05 * epoch, 0.98) for epoch in epochs]

	parallelnet_df = pd.DataFrame({"epoch": epochs, "loss": loss, "accuracy": accuracy})
	parallelnet_csv = mock_dir / "parallelnet.csv"
	parallelnet_df.to_csv(parallelnet_csv, index=False)

	pytorch_df = pd.DataFrame(
		{
			"epoch": epochs,
			"loss": [value * 1.05 for value in loss],
			"accuracy": [max(value - 0.02, 0.0) for value in accuracy],
		}
	)
	pytorch_csv = mock_dir / "pytorch.csv"
	pytorch_df.to_csv(pytorch_csv, index=False)

	plot_convergence(parallelnet_csv, output="plots/convergence.png")

	scaling_results = {
		1: {"time_per_epoch": 10.0, "final_acc": 0.82},
		2: {"time_per_epoch": 5.8, "final_acc": 0.83},
		4: {"time_per_epoch": 3.3, "final_acc": 0.84},
		8: {"time_per_epoch": 2.2, "final_acc": 0.84},
	}
	plot_scaling_efficiency(scaling_results, output="plots/scaling.png")

	comm_results = {
		1: {"compute_ms": 120.0, "sync_ms": 5.0},
		2: {"compute_ms": 70.0, "sync_ms": 12.0},
		4: {"compute_ms": 45.0, "sync_ms": 20.0},
		8: {"compute_ms": 32.0, "sync_ms": 36.0},
	}
	plot_communication_breakdown(comm_results, output="plots/comm_breakdown.png")

	plot_accuracy_parity(parallelnet_csv, pytorch_csv, output="plots/parity.png")
