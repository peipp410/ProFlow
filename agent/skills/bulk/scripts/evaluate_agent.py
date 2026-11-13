#!/usr/bin/env python3
"""Summarize routing and task outcomes from annotated Agent run logs.

Input is JSONL with one record per benchmark task. This evaluator deliberately
does not call the Agent: ``actual_route`` and ``success`` must come from the
Agent's run trace and be checked against the benchmark's expected labels.
Only the Python standard library is required.
"""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def parse_bool(value: Any, field: str, line_number: int) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "success", "succeeded"}:
            return True
        if normalized in {"false", "no", "0", "failed", "failure"}:
            return False
    raise ValueError(f"line {line_number}: {field} must be a boolean (got {value!r})")


def load_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    seen_ids = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"line {line_number}: each JSONL record must be an object")
            required = ("id", "expected_route", "actual_route", "success")
            missing = [key for key in required if key not in record]
            if missing:
                raise ValueError(f"line {line_number}: missing required field(s): {', '.join(missing)}")

            task_id = str(record["id"]).strip()
            expected = str(record["expected_route"]).strip()
            actual = str(record["actual_route"]).strip()
            if not task_id or not expected or not actual:
                raise ValueError(f"line {line_number}: id and route labels must be non-empty")
            if task_id in seen_ids:
                raise ValueError(f"line {line_number}: duplicate task id {task_id!r}")
            seen_ids.add(task_id)

            normalized = dict(record)
            normalized["id"] = task_id
            normalized["expected_route"] = expected
            normalized["actual_route"] = actual
            normalized["success"] = parse_bool(record["success"], "success", line_number)
            normalized["task_type"] = str(record.get("task_type") or "unspecified").strip()
            mode = record.get("failure_mode")
            normalized["failure_mode"] = str(mode).strip() if mode is not None else ""
            if not normalized["task_type"]:
                normalized["task_type"] = "unspecified"
            records.append(normalized)
    if not records:
        raise ValueError(f"No task records found in {path}")
    return records


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> Tuple[Optional[float], Optional[float]]:
    if total == 0:
        return None, None
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def rate_metrics(successes: int, total: int) -> Dict[str, Any]:
    low, high = wilson_interval(successes, total)
    return {
        "numerator": successes,
        "denominator": total,
        "rate": successes / total if total else None,
        "ci95_wilson": [low, high],
    }


def evaluate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    route_correct = sum(r["expected_route"] == r["actual_route"] for r in records)
    task_successes = sum(r["success"] for r in records)
    failures = [r for r in records if not r["success"]]
    failure_counts = Counter(r["failure_mode"] or "unclassified" for r in failures)

    route_pairs = Counter((r["expected_route"], r["actual_route"]) for r in records)
    routes = sorted({route for pair in route_pairs for route in pair})
    confusion = {
        expected: {actual: route_pairs[(expected, actual)] for actual in routes}
        for expected in routes
    }

    task_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        task_groups[record["task_type"]].append(record)
    by_task_type = {}
    for task_type, group in sorted(task_groups.items()):
        n_route_ok = sum(r["expected_route"] == r["actual_route"] for r in group)
        n_success = sum(r["success"] for r in group)
        by_task_type[task_type] = {
            "n": len(group),
            "routing_accuracy": rate_metrics(n_route_ok, len(group)),
            "task_success_rate": rate_metrics(n_success, len(group)),
        }

    return {
        "n_tasks": len(records),
        "routing_accuracy": rate_metrics(route_correct, len(records)),
        "task_success_rate": rate_metrics(task_successes, len(records)),
        "n_failures": len(failures),
        "failure_mode_counts": dict(sorted(failure_counts.items())),
        "failure_mode_denominator": len(failures),
        "route_labels": routes,
        "route_confusion_matrix": confusion,
        "by_task_type": by_task_type,
    }


def write_csv(path: Path, header: Iterable[str], rows: Iterable[Iterable[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_outputs(records: List[Dict[str, Any]], summary: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    routes = summary["route_labels"]
    matrix = summary["route_confusion_matrix"]
    write_csv(
        out_dir / "route_confusion.csv",
        ["expected_route"] + routes,
        ([expected] + [matrix[expected][actual] for actual in routes] for expected in routes),
    )

    by_type_rows = []
    for task_type, metrics in summary["by_task_type"].items():
        route = metrics["routing_accuracy"]
        success = metrics["task_success_rate"]
        by_type_rows.append([
            task_type, metrics["n"], route["numerator"], route["rate"],
            success["numerator"], success["rate"],
        ])
    write_csv(
        out_dir / "by_task_type.csv",
        ["task_type", "n_tasks", "routing_correct", "routing_accuracy", "successful_tasks", "task_success_rate"],
        by_type_rows,
    )

    failures = [r for r in records if not r["success"]]
    failure_counts = summary["failure_mode_counts"]
    total_failures = len(failures)
    write_csv(
        out_dir / "failure_modes.csv",
        ["failure_mode", "count", "fraction_of_failures"],
        ([mode, count, count / total_failures if total_failures else 0.0]
         for mode, count in sorted(failure_counts.items(), key=lambda item: (-item[1], item[0]))),
    )

    with (out_dir / "failure_cases.jsonl").open("w", encoding="utf-8") as handle:
        for record in failures:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_jsonl", required=True, help="Annotated Agent task traces in JSONL format")
    parser.add_argument("--out_dir", required=True, help="Directory for summary and CSV reports")
    args = parser.parse_args()
    records = load_records(Path(args.input_jsonl))
    summary = evaluate(records)
    write_outputs(records, summary, Path(args.out_dir))
    print(json.dumps({
        "n_tasks": summary["n_tasks"],
        "routing_accuracy": summary["routing_accuracy"],
        "task_success_rate": summary["task_success_rate"],
        "n_failures": summary["n_failures"],
        "out_dir": str(Path(args.out_dir)),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
