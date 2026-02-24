"""
Convert MedCalc-Bench train_data.csv → data/train.jsonl

Each output row is a complete tool-use trajectory:
  system → user → assistant(calc_info call) → tool → assistant(execute_calc call)
  → tool → assistant("Done.")

Run with:
    uv run scripts/build_training_data.py [--limit N] [--out data/train.jsonl]
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import sys
from pathlib import Path

# Allow running as a script from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.executor import run as exec_run
from src.prompts import build_system_prompt
from src.schemas import (
    MEDCALC_ID_TO_CALC_ID,
    _FIELD_DEFS,
    _SKIP_KEYS,
    load_schemas,
    public_schema,
)

MEDCALC_DIR = Path(__file__).parent.parent / "MedCalc-Bench-Verified"
TRAIN_CSV = MEDCALC_DIR / "datasets" / "train_data.csv"
_NAME_TO_PYTHON_PATH = MEDCALC_DIR / "calculator_implementations" / "name_to_python.json"


def _load_name_to_python() -> dict[str, dict]:
    with open(_NAME_TO_PYTHON_PATH) as f:
        return json.load(f)


def _parse_entities(raw: str) -> dict | None:
    try:
        val = ast.literal_eval(raw)
        return val if isinstance(val, dict) else None
    except Exception:
        return None


def _to_execute_var(value: object) -> dict:
    """Convert a MedCalc Relevant Entities value to execute_calc variable format."""
    if isinstance(value, list) and len(value) == 2:
        return {"value": value[0], "unit": str(value[1])}
    if isinstance(value, bool):
        return {"value": value}
    if isinstance(value, (int, float)):
        return {"value": value}
    return {"value": value}


def _build_mme_variables(entities: dict) -> dict:
    """
    Parse MME Relevant Entities (dynamic drug keys) into structured format.

    Entities look like:
        {'HYDROcodone Dose': [5, 'mg'], 'HYDROcodone Dose Per Day': [3, '']}

    Returns:
        {"medications": [{"drug": "HYDROcodone", "dose": 5, "dose_unit": "mg",
                          "doses_per_day": 3}]}
    """
    # Collect drug names from " Dose" keys (skip " Dose Per Day")
    drugs: dict[str, dict] = {}
    for key, value in entities.items():
        if "Dose Per Day" in key:
            name = key.replace(" Dose Per Day", "")
            drugs.setdefault(name, {})["doses_per_day"] = value[0] if isinstance(value, list) else value
        elif "Dose" in key:
            name = key.replace(" Dose", "")
            drugs.setdefault(name, {})
            if isinstance(value, list) and len(value) == 2:
                drugs[name]["dose"] = value[0]
                drugs[name]["dose_unit"] = str(value[1])
            else:
                drugs[name]["dose"] = value
                drugs[name]["dose_unit"] = "mg"

    medications = []
    for name, info in drugs.items():
        medications.append({
            "drug": name,
            "dose": info.get("dose", 0),
            "dose_unit": info.get("dose_unit", "mg"),
            "doses_per_day": info.get("doses_per_day", 1),
        })

    return {"medications": medications} if medications else {}


def _build_variables(
    entities: dict,
    medcalc_id: int,
    n2p: dict[str, dict],
    schema: dict,
) -> dict[str, dict]:
    """
    Map Relevant Entities {note_field: value} → {field_id: {"value": ..., "unit": ...}}
    using the schema's field definitions.
    """
    # MME has dynamic drug-name keys — handle separately
    if schema["calc_id"] == "mme":
        return _build_mme_variables(entities)

    entry = n2p.get(str(medcalc_id), {})
    # note_field → func_param
    param_map = {k: v for k, v in entry.items() if k not in _SKIP_KEYS}
    # func_param → field_id (from schema)
    param_to_field_id = {
        inp["function_param"]: inp["id"] for inp in schema["inputs"]
    }

    variables: dict[str, dict] = {}
    for note_field, value in entities.items():
        func_param = param_map.get(note_field)
        if func_param is None:
            continue
        field_id = param_to_field_id.get(func_param, func_param)
        variables[field_id] = _to_execute_var(value)

    return variables


def build_trajectory(
    row: dict,
    schemas: dict,
    n2p: dict,
    system_prompt: str,
) -> dict | None:
    """Build one JSONL example or return None to skip."""
    try:
        medcalc_id = int(row["Calculator ID"])
    except (KeyError, ValueError):
        return None

    calc_id = MEDCALC_ID_TO_CALC_ID.get(medcalc_id)
    if calc_id is None:
        return None

    schema = schemas.get(calc_id)
    if schema is None:
        return None

    entities = _parse_entities(row.get("Relevant Entities", ""))
    if not entities:
        return None

    variables = _build_variables(entities, medcalc_id, n2p, schema)
    if not variables:
        return None

    # Run the actual calculator to get a real execute_calc response
    exec_resp = exec_run(calc_id, variables, schemas)
    if not exec_resp["success"]:
        # Skip rows where the implementation fails (rare)
        return None

    calc_info_resp = public_schema(schema)

    id1 = str(random.randint(100_000_000, 999_999_999))
    id2 = str(random.randint(100_000_000, 999_999_999))

    user_content = f"{row['Patient Note']}\n\n{row['Question']}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
        {
            "role": "assistant",
            "tool_calls": [{
                "id": id1,
                "type": "function",
                "function": {
                    "name": "calc_info",
                    "arguments": json.dumps({"calc_id": calc_id}),
                },
            }],
        },
        {
            "role": "tool",
            "content": json.dumps(calc_info_resp),
            "tool_call_id": id1,
        },
        {
            "role": "assistant",
            "tool_calls": [{
                "id": id2,
                "type": "function",
                "function": {
                    "name": "execute_calc",
                    "arguments": json.dumps({
                        "calc_id": calc_id,
                        "variables": variables,
                    }),
                },
            }],
        },
        {
            "role": "tool",
            "content": json.dumps(exec_resp),
            "tool_call_id": id2,
        },
        {"role": "assistant", "content": "Done."},
    ]

    return {"messages": messages}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap rows processed (for quick tests)")
    parser.add_argument("--out", default="data/train.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print("Loading schemas …")
    schemas = load_schemas()
    n2p = _load_name_to_python()
    system_prompt = build_system_prompt(schemas)
    print(f"  {len(schemas)} calculators loaded")

    print(f"Reading {TRAIN_CSV} …")
    import csv
    with open(TRAIN_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if args.limit:
        rows = rows[: args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = skipped = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for i, row in enumerate(rows):
            if i % 500 == 0:
                print(f"  {i}/{len(rows)}  written={written}  skipped={skipped}")
            traj = build_trajectory(row, schemas, n2p, system_prompt)
            if traj is None:
                skipped += 1
                continue
            fout.write(json.dumps(traj, ensure_ascii=False) + "\n")
            written += 1

    print(f"\nDone. {written} examples written to {out_path}  ({skipped} skipped)")


if __name__ == "__main__":
    main()
