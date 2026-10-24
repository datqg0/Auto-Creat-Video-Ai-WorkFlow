# Deep-ML Knowledge Base: Mlops

Total verified problems: 1

Source: [Deep-ML (Open-Deep-ML)](https://www.deep-ml.com/)

---

## Problem 187: Build a Simple ETL Pipeline (MLOps)
- **Difficulty**: `medium`
- **Category**: MLOps

### Description & Spec
## Problem

Implement a simple ETL (Extract-Transform-Load) pipeline for model-ready data preparation.

Given a CSV-like string containing user events with columns: `user_id,event_type,value` (header included), write a function `run_etl(csv_text)` that:

1. Extracts rows from the raw CSV text.
2. Transforms data by:
	- Filtering only rows where `event_type == "purchase"`.
	- Converting `value` to float and dropping invalid rows.
	- Aggregating total purchase `value` per `user_id`.
3. Loads the transformed results by returning a list of `(user_id, total_value)` tuples sorted by `user_id` ascending.

Assume small inputs (no external libs), handle extra whitespace, and ignore blank lines.

### Mathematical Derivation & Intuition
## Solution Explanation

This task mirrors a minimal MLOps ETL flow that prepares data for downstream modeling.

### ETL breakdown
- Extract: parse raw CSV text, ignore blanks, and split into header and rows.
- Transform:
	- Filter only relevant records (event_type == "purchase").
	- Cast `value` to float; discard invalid rows to maintain data quality.
	- Aggregate total purchase value per user to create compact features.
- Load: return a deterministic, sorted list of `(user_id, total_value)`.

### Why this design?
- Input sanitation prevents runtime errors and poor-quality features.
- Aggregation compresses event-level logs into user-level features commonly used in models.
- Sorting produces stable, testable outputs.

### Complexity
- For N rows, parsing and aggregation run in O(N); sorting unique users U costs O(U log U).

### Extensions
- Add schema validation and logging.
- Write outputs to files or databases.
- Schedule ETL runs and add monitoring for drift and freshness.

### Verified Pure NumPy Solution (From Scratch)
```python
from typing import List, Tuple


def run_etl(csv_text: str) -> List[Tuple[str, float]]:
	"""Reference ETL implementation.

	- Extract: parse CSV text, skip header, strip whitespace, ignore blanks
	- Transform: keep event_type == "purchase"; parse value as float; aggregate per user
	- Load: return sorted list of (user_id, total_value) by user_id asc
	"""
	lines = [line.strip() for line in csv_text.splitlines() if line.strip()]
	if not lines:
		return []
	# header
	header = lines[0]
	rows = lines[1:]

	# indices from header (allow varying order and case)
	headers = [h.strip().lower() for h in header.split(",")]
	try:
		idx_user = headers.index("user_id")
		idx_event = headers.index("event_type")
		idx_value = headers.index("value")
	except ValueError:
		# header missing required columns
		return []

	aggregates: dict[str, float] = {}
	for row in rows:
		parts = [c.strip() for c in row.split(",")]
		if len(parts) <= max(idx_user, idx_event, idx_value):
			continue
		user_id = parts[idx_user]
		event_type = parts[idx_event].lower()
		if event_type != "purchase":
			continue
		try:
			value = float(parts[idx_value])
		except ValueError:
			continue
		aggregates[user_id] = aggregates.get(user_id, 0.0) + value

	return sorted(aggregates.items(), key=lambda kv: kv[0])
```

### Verified Sample Trace Data
```json
{
  "test": "print(run_etl('user_id,event_type,value\\n u1, purchase, 10.0\\n u2, view, 1.0\\n u1, purchase, 5\\n u3, purchase, not_a_number\\n u2, purchase, 3.5 \\n'))",
  "expected_output": "[('u1', 15.0), ('u2', 3.5)]"
}
```

---

