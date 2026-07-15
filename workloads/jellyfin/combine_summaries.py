#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--output", default="all-runs.csv")
    args = parser.parse_args()

    files = sorted(Path(args.results).glob("*/summary.csv"))
    rows: list[dict[str, str]] = []

    for path in files:
        with path.open(newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))

    if not rows:
        raise SystemExit(f"No summary.csv files found under {args.results!r}")

    fields = sorted({key for row in rows for key in row})
    with Path(args.output).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} runs to {args.output}")


if __name__ == "__main__":
    main()
