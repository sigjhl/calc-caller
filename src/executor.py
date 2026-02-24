"""
Wraps MedCalc-Bench Python implementations to produce execute_calc responses.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

from src.schemas import MEDCALC_DIR

_IMPL_DIR = MEDCALC_DIR / "calculator_implementations"

# Add once at import time; calculators import each other so it must persist.
if str(_IMPL_DIR) not in sys.path:
    sys.path.insert(0, str(_IMPL_DIR))


def _to_medcalc(value: Any, unit: str | None, fmt: str) -> Any:
    """Convert an execute_calc variable value to the format MedCalc expects."""
    if fmt == "list":
        # MedCalc expects [numeric_value, unit_string]
        u = unit or ""
        return [value, u]
    if fmt == "boolean":
        return bool(value)
    # "scalar" or "text": pass through as-is
    return value


def _build_mme_input(medications: list[dict]) -> dict[str, Any]:
    """
    Convert structured medications list to the flat dict mme.py expects.

    Model sends:
        [{"drug": "HYDROcodone", "dose": 5, "dose_unit": "mg", "doses_per_day": 3}]

    MedCalc expects:
        {"HYDROcodone Dose": [5, "mg"], "HYDROcodone Dose Per Day": [3, ""]}
    """
    flat: dict[str, Any] = {}
    for med in medications:
        name = med["drug"]
        dose = med.get("dose", 0)
        dose_unit = med.get("dose_unit", "mg")
        per_day = med.get("doses_per_day", 1)
        flat[f"{name} Dose"] = [dose, dose_unit]
        flat[f"{name} Dose Per Day"] = [per_day, ""]
    return flat


def run(calc_id: str, variables: dict[str, Any],
        schemas: dict[str, Any]) -> dict[str, Any]:
    """
    Execute a calculator.

    Parameters
    ----------
    calc_id   : the calculator identifier
    variables : {field_id: {"value": ..., "unit": ...}} as produced by the model
    schemas   : full schema dict from load_schemas()

    Returns
    -------
    execute_calc response dict
    """
    schema = schemas.get(calc_id)
    if schema is None:
        return {
            "success": False, "outputs": None,
            "errors": [f"Unknown calculator: {calc_id}"],
            "warnings": [], "audit_trace": None,
        }

    warnings: list[str] = []

    # ── MME special path (dynamic drug inputs) ────────────────────────────
    if calc_id == "mme":
        meds = variables.get("medications")
        if not meds or not isinstance(meds, list):
            return {
                "success": False, "outputs": None,
                "errors": ["'medications' array is required for MME calculator."],
                "warnings": warnings, "audit_trace": None,
            }
        input_variables = _build_mme_input(meds)
        inputs_used = {
            m["drug"]: f"{m.get('dose',0)} {m.get('dose_unit','mg')} × {m.get('doses_per_day',1)}/day"
            for m in meds
        }
    else:
        # ── Standard path ─────────────────────────────────────────────────
        field_map: dict[str, dict] = {inp["id"]: inp for inp in schema["inputs"]}

        input_variables: dict[str, Any] = {}
        inputs_used: dict[str, str] = {}
        errors: list[str] = []

        for field_id, value_obj in variables.items():
            if field_id not in field_map:
                warnings.append(f"Unrecognised field ignored: {field_id}")
                continue

            inp = field_map[field_id]

            if isinstance(value_obj, dict):
                raw_value = value_obj.get("value")
                unit = value_obj.get("unit") or inp.get("canonical_unit") or ""
            else:
                raw_value = value_obj
                unit = inp.get("canonical_unit") or ""

            if raw_value is None:
                errors.append(f"Null value for required field: {field_id}")
                continue

            converted = _to_medcalc(raw_value, unit, inp["medcalc_fmt"])
            input_variables[inp["function_param"]] = converted
            inputs_used[field_id] = f"{raw_value} {unit}".strip()

        if errors:
            return {
                "success": False, "outputs": None,
                "errors": errors, "warnings": warnings, "audit_trace": None,
            }

    # Import the module and call the explanation function
    module_name = Path(schema["file_path"]).stem
    try:
        module = importlib.import_module(module_name)
        func = getattr(module, schema["function"])
        result = func(input_variables)
        answer = result.get("Answer", result)
    except Exception as exc:
        return {
            "success": False, "outputs": None,
            "errors": [f"Calculation error: {exc}"],
            "warnings": warnings, "audit_trace": None,
        }

    return {
        "success": True,
        "outputs": {"result": answer},
        "errors": [],
        "warnings": warnings,
        "audit_trace": {
            "inputs_used": inputs_used,
            "log": [f"Computed {schema['title']}."],
        },
    }
