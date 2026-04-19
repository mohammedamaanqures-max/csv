from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

DATASET_PATH = Path("data/airline_passenger_satisfaction.csv")


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def safe_average(values: list[int]) -> float:
    return round(sum(values) / len(values), 2) if values else 0.0


def safe_int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def run_datamining(rows: list[dict[str, str]]) -> dict[str, object]:
    total = len(rows)
    satisfied = [r for r in rows if r["Satisfaction"] == "satisfied"]

    satisfaction_rate = round((len(satisfied) / total) * 100, 2) if total else 0.0

    class_breakdown: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "satisfied": 0})
    for row in rows:
        ticket_class = row["Class"]
        class_breakdown[ticket_class]["total"] += 1
        if row["Satisfaction"] == "satisfied":
            class_breakdown[ticket_class]["satisfied"] += 1

    class_satisfaction = {
        cls: round((stats["satisfied"] / stats["total"]) * 100, 2) if stats["total"] else 0.0
        for cls, stats in class_breakdown.items()
    }

    satisfied_delays = [
        delay
        for delay in (safe_int(r["ArrivalDelayMinutes"]) for r in satisfied)
        if delay is not None
    ]
    unsatisfied_delays = [
        delay
        for delay in (safe_int(r["ArrivalDelayMinutes"]) for r in rows if r["Satisfaction"] != "satisfied")
        if delay is not None
    ]
    avg_delay_satisfied = safe_average(satisfied_delays)
    avg_delay_unsatisfied = safe_average(unsatisfied_delays)

    travel_type_counter = Counter(r["TypeOfTravel"] for r in satisfied)

    return {
        "records": total,
        "satisfaction_rate": satisfaction_rate,
        "class_satisfaction": class_satisfaction,
        "avg_arrival_delay_satisfied": avg_delay_satisfied,
        "avg_arrival_delay_unsatisfied": avg_delay_unsatisfied,
        "top_satisfied_travel_type": travel_type_counter.most_common(1)[0][0]
        if travel_type_counter
        else None,
    }


def main() -> None:
    rows = load_rows(DATASET_PATH)
    results = run_datamining(rows)

    print("=== Airline Passenger Satisfaction: Data Mining Summary ===")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Total records: {results['records']}")
    print(f"Overall satisfaction rate: {results['satisfaction_rate']}%")
    print("Satisfaction rate by class:")
    for cls, rate in sorted(results["class_satisfaction"].items()):
        print(f"  - {cls}: {rate}%")
    print(f"Avg arrival delay (satisfied): {results['avg_arrival_delay_satisfied']} minutes")
    print(f"Avg arrival delay (not satisfied): {results['avg_arrival_delay_unsatisfied']} minutes")
    print(f"Top travel type among satisfied passengers: {results['top_satisfied_travel_type']}")


if __name__ == "__main__":
    main()
