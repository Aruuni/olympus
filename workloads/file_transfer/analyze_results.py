#!/usr/bin/env python3
"""Combine file-transfer runs and calculate flow-completion-time averages."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent


def as_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def is_completed(row: dict[str, str]) -> bool:
    return row.get("completed", "").strip().lower() in {"true", "1", "yes"}


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def rounded(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def group_fields(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row.get("experiment", ""),
        row.get("cc", ""),
        row.get("n", ""),
        row.get("file_name", ""),
        row.get("base_url", ""),
        row.get("stagger_seconds", ""),
        row.get("timeout_seconds", ""),
    )


def aggregate_row(
    scope: str,
    key: tuple[str, ...],
    rows: list[dict[str, str]],
    flow_id: str = "",
) -> dict[str, str | int]:
    completed_rows = [row for row in rows if is_completed(row)]
    fcts = [
        value
        for row in completed_rows
        if (value := as_float(row.get("fct_seconds"))) is not None
    ]
    goodputs = [
        value
        for row in completed_rows
        if (value := as_float(row.get("mean_goodput_mbps"))) is not None
    ]
    sizes = [
        value
        for row in completed_rows
        if (value := as_float(row.get("size_bytes"))) is not None
    ]
    run_count = len({row.get("experiment_id", "") for row in rows})
    timed_out = sum(row.get("status") == "timed_out" for row in rows)
    completed_count = len(completed_rows)
    sample_count = len(rows)
    experiment, cc, n, file_name, base_url, stagger, timeout = key

    return {
        "scope": scope,
        "experiment": experiment,
        "cc": cc,
        "n": n,
        "flow_id": flow_id,
        "file_name": file_name,
        "base_url": base_url,
        "stagger_seconds": stagger,
        "timeout_seconds": timeout,
        "run_count": run_count,
        "sample_count": sample_count,
        "completed_count": completed_count,
        "timeout_count": timed_out,
        "failed_count": sample_count - completed_count - timed_out,
        "completion_rate": rounded(completed_count / sample_count),
        "mean_fct_seconds": rounded(statistics.fmean(fcts) if fcts else None),
        "stdev_fct_seconds": rounded(
            statistics.stdev(fcts) if len(fcts) > 1 else (0.0 if fcts else None)
        ),
        "median_fct_seconds": rounded(statistics.median(fcts) if fcts else None),
        "p95_fct_seconds": rounded(percentile(fcts, 0.95)),
        "min_fct_seconds": rounded(min(fcts) if fcts else None),
        "max_fct_seconds": rounded(max(fcts) if fcts else None),
        "mean_goodput_mbps": rounded(
            statistics.fmean(goodputs) if goodputs else None
        ),
        "mean_size_mib": rounded(
            statistics.fmean(sizes) / (1024 * 1024) if sizes else None
        ),
    }


def write_rows(path: Path, rows: Iterable[dict[str, object]]) -> int:
    materialized = list(rows)
    if not materialized:
        raise SystemExit(f"No rows available for {path}")
    fields = list(materialized[0])
    extra_fields = sorted(
        {field for row in materialized for field in row}.difference(fields)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + extra_fields)
        writer.writeheader()
        writer.writerows(materialized)
    return len(materialized)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Combine all transfer summaries and average completed FCTs. "
            "Timeouts are reported but excluded from FCT averages."
        )
    )
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--combined", type=Path, default=ROOT / "analysis" / "combined_flows.csv"
    )
    parser.add_argument(
        "--averages", type=Path, default=ROOT / "analysis" / "fct_averages.csv"
    )
    args = parser.parse_args()

    summary_files = sorted(args.data.glob("*/*/summary.csv"))
    combined: list[dict[str, str]] = []
    for summary_path in summary_files:
        with summary_path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                row["source_summary"] = str(summary_path.resolve())
                combined.append(row)

    if not combined:
        raise SystemExit(f"No summary.csv files found below {args.data}")

    combined.sort(
        key=lambda row: (
            row.get("cc", ""),
            row.get("experiment_id", ""),
            int(row.get("flow_id", "0")),
        )
    )
    combined_count = write_rows(args.combined, combined)

    overall_groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    per_flow_groups: dict[
        tuple[tuple[str, ...], str], list[dict[str, str]]
    ] = defaultdict(list)
    for row in combined:
        key = group_fields(row)
        overall_groups[key].append(row)
        per_flow_groups[(key, row.get("flow_id", ""))].append(row)

    aggregates: list[dict[str, str | int]] = []
    for key, rows in sorted(overall_groups.items()):
        aggregates.append(aggregate_row("all_flows", key, rows))
    for (key, flow_id), rows in sorted(
        per_flow_groups.items(),
        key=lambda item: (item[0][0], int(item[0][1] or 0)),
    ):
        aggregates.append(aggregate_row("flow_position", key, rows, flow_id))

    aggregate_count = write_rows(args.averages, aggregates)
    completed_count = sum(is_completed(row) for row in combined)
    print(f"Wrote {combined_count} flow records to {args.combined}")
    print(f"Wrote {aggregate_count} aggregate rows to {args.averages}")
    print(
        f"Completed transfers: {completed_count}/{combined_count}; "
        "incomplete transfers are excluded from FCT averages."
    )


if __name__ == "__main__":
    main()
