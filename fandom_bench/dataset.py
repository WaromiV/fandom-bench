import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class BenchmarkRecord:
    """One row from the benchmark CSV."""

    row_id: str
    prompt: str
    expected_answer: Optional[str]
    metadata: Dict[str, str]


def load_dataset(
    csv_path: Path,
    prompt_column: str = "prompt",
    expected_column: str = "expected_answer",
    max_records: Optional[int] = None,
) -> List[BenchmarkRecord]:
    """Load benchmark prompts from CSV."""
    records: List[BenchmarkRecord] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if prompt_column not in reader.fieldnames:
            raise ValueError(f"CSV missing prompt column '{prompt_column}'")
        if expected_column not in reader.fieldnames:
            raise ValueError(f"CSV missing expected column '{expected_column}'")
        for row in reader:
            row_id = row.get("id") or row.get("row_id") or ""
            if not row_id:
                raise ValueError("CSV row missing id/row_id column")
            prompt = row[prompt_column]
            expected = row.get(expected_column) or None
            metadata = {
                k: v for k, v in row.items() if k not in {prompt_column, expected_column, "id", "row_id"}
            }
            records.append(
                BenchmarkRecord(
                    row_id=row_id,
                    prompt=prompt,
                    expected_answer=expected,
                    metadata=metadata,
                )
            )
            if max_records and len(records) >= max_records:
                break
    return records
