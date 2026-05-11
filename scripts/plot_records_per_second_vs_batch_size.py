#!/usr/bin/env python3
"""Plot records/second from benchmark JSON output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


_SCRIPT_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_REPO_ROOT))


NumericSeries = Dict[str, List[float]]
OverlaySeries = Dict[str, List[float] | List[str]]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot records/s from benchmark_embedding_batching_strategies JSON.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=_SCRIPT_REPO_ROOT / "benchmark_embedding_batching_strategies.json",
        help="Benchmark JSON path.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model key to plot. Defaults to the only model in the file, or the first sorted key.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_SCRIPT_REPO_ROOT / "records_per_second_vs_batch_size.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--plot",
        choices=["fixed", "token-budget", "combined"],
        default="fixed",
        help="Plot fixed batch sizes, token budgets, or both side by side. Default: fixed.",
    )
    parser.add_argument(
        "--include-unsorted",
        action="store_true",
        help="Include unsorted settings if present. Default plots sorted settings only.",
    )
    return parser.parse_args()


def _resolve_model_name(payload: Dict[str, Any], requested: str | None) -> str:
    models = payload.get("models")
    if not isinstance(models, dict) or not models:
        raise SystemExit("Benchmark JSON does not contain any models.")
    if requested is not None:
        if requested not in models:
            available = ", ".join(sorted(str(name) for name in models.keys()))
            raise SystemExit(f"Unknown model {requested!r}. Available: {available}")
        return requested
    names = sorted(str(name) for name in models.keys())
    return names[0]


def _collect_series(
    model_payload: Dict[str, Any],
    *,
    include_unsorted: bool,
    kind: str,
    value_key: str,
) -> Dict[str, NumericSeries]:
    series: Dict[str, NumericSeries] = {}
    for length_limit_key, length_payload in model_payload.items():
        if not isinstance(length_payload, dict):
            continue
        runs = length_payload.get("runs")
        if not isinstance(runs, dict):
            continue

        x_values: List[int] = []
        means: List[float] = []
        stdevs: List[float] = []

        for run_key, run_payload in runs.items():
            if not isinstance(run_payload, dict):
                continue
            setting = run_payload.get("setting")
            summary = run_payload.get("summary")
            if not isinstance(setting, dict) or not isinstance(summary, dict):
                continue
            if setting.get("kind") != kind:
                continue
            if not include_unsorted and not bool(setting.get("sort_by_length")):
                continue

            x_value = setting.get(value_key)
            mean_records = summary.get("mean_records_per_second")
            stdev_records = summary.get("stdev_records_per_second", 0.0)
            if not isinstance(x_value, int):
                continue
            if not isinstance(mean_records, (int, float)):
                continue
            if not isinstance(stdev_records, (int, float)):
                stdev_records = 0.0

            x_values.append(int(x_value))
            means.append(float(mean_records))
            stdevs.append(float(stdev_records))

        if x_values:
            order = sorted(range(len(x_values)), key=lambda index: x_values[index])
            series[str(length_limit_key)] = {
                "x_values": [x_values[index] for index in order],
                "means": [means[index] for index in order],
                "stdevs": [stdevs[index] for index in order],
            }
    if not series:
        mode_text = "sorted or unsorted" if include_unsorted else "sorted"
        raise SystemExit(f"No {kind} {mode_text} runs were found for this model.")
    return series


def _collect_realized_batch_series(
    model_payload: Dict[str, Any],
    *,
    include_unsorted: bool,
) -> Dict[str, OverlaySeries]:
    series: Dict[str, OverlaySeries] = {}
    for length_limit_key, length_payload in model_payload.items():
        if not isinstance(length_payload, dict):
            continue
        runs = length_payload.get("runs")
        if not isinstance(runs, dict):
            continue

        x_values: List[float] = []
        means: List[float] = []
        stdevs: List[float] = []
        labels: List[str] = []

        for run_payload in runs.values():
            if not isinstance(run_payload, dict):
                continue
            setting = run_payload.get("setting")
            summary = run_payload.get("summary")
            if not isinstance(setting, dict) or not isinstance(summary, dict):
                continue
            if setting.get("kind") != "token_budget":
                continue
            if not include_unsorted and not bool(setting.get("sort_by_length")):
                continue

            realized_batch_size = summary.get("mean_realized_batch_size")
            mean_records = summary.get("mean_records_per_second")
            stdev_records = summary.get("stdev_records_per_second", 0.0)
            token_budget = setting.get("max_batch_tokens")
            if not isinstance(realized_batch_size, (int, float)):
                continue
            if not isinstance(mean_records, (int, float)):
                continue
            if not isinstance(stdev_records, (int, float)):
                stdev_records = 0.0

            x_values.append(float(realized_batch_size))
            means.append(float(mean_records))
            stdevs.append(float(stdev_records))
            labels.append(str(token_budget) if isinstance(token_budget, int) else "token-budget")

        if x_values:
            order = sorted(range(len(x_values)), key=lambda index: x_values[index])
            series[str(length_limit_key)] = {
                "x_values": [x_values[index] for index in order],
                "means": [means[index] for index in order],
                "stdevs": [stdevs[index] for index in order],
                "labels": [labels[index] for index in order],
            }
    return series


def _display_length_limit(key: str) -> str:
    return "No length limit" if key == "none" else f"Max length {key}"


def _format_tick(value: float) -> str:
    rounded = round(float(value), 1)
    if abs(rounded - round(rounded)) < 1e-9:
        return str(int(round(rounded)))
    return f"{rounded:.1f}"


def _plot_panel(
    ax: Any,
    *,
    series: Dict[str, NumericSeries],
    x_label: str,
    title: str,
    overlay_series: Dict[str, OverlaySeries] | None = None,
) -> None:
    xticks = set()
    for length_limit, values in sorted(series.items(), key=lambda item: (item[0] != "none", item[0])):
        line = ax.errorbar(
            values["x_values"],
            values["means"],
            yerr=values["stdevs"],
            marker="o",
            linewidth=2,
            capsize=4,
            label=_display_length_limit(length_limit),
        )
        xticks.update(float(x_value) for x_value in values["x_values"])

        if overlay_series is None:
            continue
        overlay_values = overlay_series.get(length_limit)
        if overlay_values is None:
            continue
        color = line.lines[0].get_color()
        ax.errorbar(
            overlay_values["x_values"],
            overlay_values["means"],
            yerr=overlay_values["stdevs"],
            marker="s",
            linestyle="none",
            color=color,
            capsize=4,
        )
        for x_value, y_value, label in zip(
            overlay_values["x_values"],
            overlay_values["means"],
            overlay_values["labels"],
        ):
            ax.annotate(
                label,
                (x_value, y_value),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
                color=color,
            )
        xticks.update(float(x_value) for x_value in overlay_values["x_values"])

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Records / second")
    sorted_xticks = sorted(xticks)
    ax.set_xticks(sorted_xticks)
    ax.set_xticklabels([_format_tick(x_value) for x_value in sorted_xticks])
    ax.grid(True, alpha=0.3)


def main() -> None:
    args = _parse_args()
    payload = json.loads(args.json.read_text(encoding="utf-8"))

    model_name = _resolve_model_name(payload, args.model)
    model_payload_obj = payload["models"][model_name]
    if not isinstance(model_payload_obj, dict):
        raise SystemExit(f"Model payload for {model_name!r} is not a mapping.")

    device = payload.get("device", "unknown-device")
    dtype = payload.get("dtype", "default")

    if args.plot == "fixed":
        fixed_series = _collect_series(
            model_payload_obj.get("settings", {}),
            include_unsorted=args.include_unsorted,
            kind="fixed",
            value_key="batch_size",
        )
        realized_token_series = _collect_realized_batch_series(
            model_payload_obj.get("settings", {}),
            include_unsorted=args.include_unsorted,
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        _plot_panel(
            ax,
            series=fixed_series,
            x_label="Batch size",
            title=f"{model_name}: records/s vs batch size",
            overlay_series=realized_token_series,
        )
        color_handles, color_labels = ax.get_legend_handles_labels()
        style_handles = [
            Line2D([0], [0], color="black", marker="o", linewidth=2, label="Fixed batch"),
            Line2D([0], [0], color="black", marker="s", linestyle="none", label="Token budget (realized batch)"),
        ]
        first_legend = ax.legend(color_handles, color_labels, loc="upper right", frameon=False)
        ax.add_artist(first_legend)
        ax.legend(handles=style_handles, loc="lower left", frameon=False)
    elif args.plot == "token-budget":
        token_series = _collect_series(
            model_payload_obj.get("settings", {}),
            include_unsorted=args.include_unsorted,
            kind="token_budget",
            value_key="max_batch_tokens",
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        _plot_panel(
            ax,
            series=token_series,
            x_label="Max batch tokens",
            title=f"{model_name}: records/s vs token budget",
        )
        ax.legend(frameon=False)
    else:
        fixed_series = _collect_series(
            model_payload_obj.get("settings", {}),
            include_unsorted=args.include_unsorted,
            kind="fixed",
            value_key="batch_size",
        )
        realized_token_series = _collect_realized_batch_series(
            model_payload_obj.get("settings", {}),
            include_unsorted=args.include_unsorted,
        )
        token_series = _collect_series(
            model_payload_obj.get("settings", {}),
            include_unsorted=args.include_unsorted,
            kind="token_budget",
            value_key="max_batch_tokens",
        )
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        _plot_panel(
            axes[0],
            series=fixed_series,
            x_label="Batch size",
            title="Fixed batch sizes",
            overlay_series=realized_token_series,
        )
        _plot_panel(
            axes[1],
            series=token_series,
            x_label="Max batch tokens",
            title="Token-budget batching",
        )
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.96),
            ncol=max(1, len(labels)),
            frameon=False,
        )
        style_handles = [
            Line2D([0], [0], color="black", marker="o", linewidth=2, label="Fixed batch"),
            Line2D([0], [0], color="black", marker="s", linestyle="none", label="Token budget (realized batch)"),
        ]
        axes[0].legend(handles=style_handles, loc="lower left", frameon=False)

    fig.suptitle(f"{model_name}: records/s batching comparison\n{device}, dtype={dtype}", y=0.995)
    if args.plot == "combined":
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    else:
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)

    print(args.output)


if __name__ == "__main__":
    main()