#!/usr/bin/env python3
"""
Local MedCalc-Bench evaluation — runs the fine-tuned model directly against
OmniCalc calculator backends.  No LM Studio / no server needed.

Usage (from repo root):
    uv run python eval/eval_local.py
    uv run python eval/eval_local.py --calc-filter creatinine_clearance
    uv run python eval/eval_local.py --limit 20
    uv run python eval/eval_local.py --output eval/results/eval_results.json
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add parent directory to path so we can import src/ and train modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.prompts import TOOL_SCHEMAS, build_system_prompt
from train import _build_tool_block, format_to_gemma

# ── calculator name → calc_id mapping (from benchmark.py) ───────────────────
CALC_NAME_TO_ID = {
    "APACHE II Score": "apache2",
    "Adjusted Body Weight": "adjusted_body_weight",
    "Albumin Corrected Anion Gap": "albumin_corrected_anion_gap",
    "Albumin Corrected Delta Gap": "albumin_corrected_delta_gap",
    "Albumin Corrected Delta Ratio": "albumin_corrected_delta_ratio",
    "Anion Gap": "anion_gap",
    "Body Mass Index (BMI)": "bmi",
    "Body Surface Area Calculator": "bsa",
    "CHA2DS2-VASc Score for Atrial Fibrillation Stroke Risk": "cha2ds2_vasc",
    "CKD-EPI Equations for Glomerular Filtration Rate": "ckd_epi_gfr",
    "CURB-65 Score for Pneumonia Severity": "curb65",
    "Calcium Correction for Hypoalbuminemia": "calcium_correction",
    "Caprini Score for Venous Thromboembolism (2005)": "caprini_vte",
    "Centor Score (Modified/McIsaac) for Strep Pharyngitis": "centor_score",
    "Charlson Comorbidity Index (CCI)": "cci",
    "Child-Pugh Score for Cirrhosis Mortality": "child_pugh",
    "Creatinine Clearance (Cockcroft-Gault Equation)": "creatinine_clearance",
    "Delta Gap": "delta_gap",
    "Delta Ratio": "delta_ratio",
    "Estimated Date of Conception": "estimated_conception_date",
    "Estimated Due Date": "estimated_due_date",
    "Estimated Gestational Age": "estimated_gestational_age",
    "FeverPAIN Score for Strep Pharyngitis": "feverpain",
    "Fibrosis-4 (FIB-4) Index for Liver Fibrosis": "fib4",
    "Fractional Excretion of Sodium (FENa)": "fena",
    "Framingham Risk Score for Hard Coronary Heart Disease": "framingham_chd",
    "Free Water Deficit": "free_water_deficit",
    "Glasgow Coma Score (GCS)": "glasgow_coma_score",
    "Glasgow-Blatchford Bleeding Score (GBS)": "glasgow_blatchford",
    "HAS-BLED Score for Major Bleeding Risk": "has_bled",
    "HEART Score for Major Cardiac Events": "heart_score",
    "HOMA-IR (Homeostatic Model Assessment for Insulin Resistance)": "homa_ir",
    "Ideal Body Weight": "ideal_body_weight",
    "LDL Calculated": "ldl_calculated",
    "MDRD GFR Equation": "mdrd_gfr",
    "MELD Na (UNOS/OPTN)": "meld_na",
    "Maintenance Fluids Calculations": "maintenance_fluids",
    "Mean Arterial Pressure (MAP)": "mean_arterial_pressure",
    "Morphine Milligram Equivalents (MME) Calculator": "mme",
    "PERC Rule for Pulmonary Embolism": "perc_rule",
    "PSI Score: Pneumonia Severity Index for CAP": "psi_cap",
    "QTc Bazett Calculator": "qtc_bazett",
    "QTc Framingham Calculator": "qtc_framingham",
    "QTc Fridericia Calculator": "qtc_fridericia",
    "QTc Hodges Calculator": "qtc_hodges",
    "QTc Rautaharju Calculator": "qtc_rautaharju",
    "Revised Cardiac Risk Index for Pre-Operative Risk": "rcri",
    "SIRS Criteria": "sirs",
    "Sequential Organ Failure Assessment (SOFA) Score": "sofa",
    "Serum Osmolality": "serum_osmolality",
    "Sodium Correction for Hyperglycemia": "sodium_correction_hyperglycemia",
    "Steroid Conversion Calculator": "steroid_conversion",
    "Target weight": "target_weight",
    "Wells' Criteria for DVT": "wells_dvt",
    "Wells' Criteria for Pulmonary Embolism": "wells_pe",
}

MAX_TOOL_TURNS = 6  # safety limit: calc_info + execute_calc + maybe a retry


# ── result checking (ported from benchmark.py) ──────────────────────────────

def parse_result_value(result: Any) -> Optional[float]:
    if result is None:
        return None
    if isinstance(result, (int, float)):
        return float(result)
    if isinstance(result, str):
        try:
            return float(result)
        except ValueError:
            pass
        # Gestational age: "X weeks and Y days"
        ga = re.match(r"(\d+)\s*weeks?\s*(?:and\s*)?(\d+)\s*days?", result, re.IGNORECASE)
        if ga:
            return float(int(ga.group(1)) * 7 + int(ga.group(2)))
        m = re.search(r"[-+]?\d*\.?\d+", result)
        if m:
            return float(m.group())
    if isinstance(result, (list, tuple)) and len(result) == 2:
        try:
            w = re.search(r"\d+", str(result[0]))
            d = re.search(r"\d+", str(result[1]))
            if w and d:
                return float(int(w.group()) * 7 + int(d.group()))
        except (ValueError, IndexError):
            pass
    return None


def check_result(
    result_value: Any,
    ground_truth: str,
    lower_limit: str,
    upper_limit: str,
    output_type: str,
    calc_name: str,
) -> dict:
    gt = ground_truth.strip()

    # Gestational age (must come before generic date check)
    if "Gestational Age" in calc_name:
        gt_weeks = re.search(r"(\d+)\s*weeks?", gt)
        gt_days = re.search(r"(\d+)\s*days?", gt)
        gt_val = (int(gt_weeks.group(1)) * 7 + int(gt_days.group(1))) if gt_weeks and gt_days else None
        res_val = parse_result_value(result_value)
        if res_val is None or gt_val is None:
            return {"correct": False, "result": str(result_value), "ground_truth": gt, "comparison": "parse_error"}
        try:
            lo, hi = float(lower_limit), float(upper_limit)
        except ValueError:
            lo, hi = gt_val - 0.5, gt_val + 0.5
        ok = lo <= res_val <= hi
        return {"correct": ok, "result": res_val, "ground_truth": gt_val,
                "lower": lo, "upper": hi, "comparison": "in_range" if ok else "out_of_range"}

    # Date outputs — exact match
    if output_type == "date":
        rs = str(result_value).strip() if result_value is not None else ""
        ok = rs == gt
        return {"correct": ok, "result": rs, "ground_truth": gt,
                "comparison": "exact_match" if ok else "mismatch"}

    # Numeric
    res_val = parse_result_value(result_value)
    if res_val is None:
        return {"correct": False, "result": str(result_value), "ground_truth": gt, "comparison": "parse_error"}
    try:
        lo, hi = float(lower_limit), float(upper_limit)
    except ValueError:
        try:
            gv = float(gt)
            lo, hi = gv * 0.95, gv * 1.05
        except ValueError:
            return {"correct": False, "result": res_val, "ground_truth": gt, "comparison": "limit_parse_error"}
    ok = lo <= res_val <= hi
    return {"correct": ok, "result": res_val, "ground_truth": gt,
            "lower": lo, "upper": hi, "comparison": "in_range" if ok else "out_of_range"}


# ── data loading ─────────────────────────────────────────────────────────────

def load_test_data(csv_path: str, calc_filter: Optional[str] = None) -> list[dict]:
    cases = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = CALC_NAME_TO_ID.get(row["Calculator Name"])
            if calc_filter and cid != calc_filter:
                continue
            cases.append({
                "row_number": row["Row Number"],
                "calc_name": row["Calculator Name"],
                "calc_id": cid,
                "output_type": row["Output Type"],
                "patient_note": row["Patient Note"],
                "question": row["Question"],
                "ground_truth": row["Ground Truth Answer"],
                "lower_limit": row["Lower Limit"],
                "upper_limit": row["Upper Limit"],
            })
    return cases


# ── model inference loop ─────────────────────────────────────────────────────

def build_prompt_prefix(system_prompt: str, user_msg: str) -> str:
    """Build the initial prompt (system + user + generation start)."""
    tool_block = _build_tool_block()
    return (
        f"<bos><start_of_turn>system\n{system_prompt}\n\n{tool_block}<end_of_turn>\n"
        f"<start_of_turn>user\n{user_msg}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )


def append_tool_turn(prompt: str, tool_call: dict, tool_response: str) -> str:
    """Append an assistant tool-call + tool response, then open a new model turn."""
    name = tool_call["name"]
    args = json.dumps(tool_call["arguments"])
    return (
        prompt
        + f'[TOOL_REQUEST]{{"name": "{name}", "arguments": {args}}}[END_TOOL_REQUEST]<end_of_turn>\n'
        + f"<start_of_turn>user\n{tool_response}<end_of_turn>\n"
        + f"<start_of_turn>model\n"
    )


def generate_turn(
    model, tokenizer, prompt: str, stop_ids: list[int], max_new: int = 512,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new, do_sample=False, eos_token_id=stop_ids,
        )
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)


def parse_tool_call(text: str) -> Optional[dict]:
    m = re.search(r"\[TOOL_REQUEST\](.*?)\[END_TOOL_REQUEST\]", text)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


async def run_one_case(
    model,
    tokenizer,
    tool_handler,
    system_prompt: str,
    patient_note: str,
    question: str,
    stop_ids: list[int],
) -> dict:
    """
    Run a single test case through the full tool-use loop.

    Returns {"success": bool, "result": Any, "turns": int, "error": str|None,
             "tool_calls": list, "final_text": str}
    """
    user_msg = f"{patient_note}\n\n{question}"
    prompt = build_prompt_prefix(system_prompt, user_msg)
    tool_calls_log = []

    for turn in range(MAX_TOOL_TURNS):
        raw = generate_turn(model, tokenizer, prompt, stop_ids)
        tc = parse_tool_call(raw)

        if tc is None:
            # Model produced a text response (e.g. "Done." or an error)
            clean = raw.replace("<end_of_turn>", "").strip()
            # If we already have a successful execute_calc result, this is fine
            return {
                "success": len(tool_calls_log) > 0 and any(
                    t.get("response", {}).get("success") for t in tool_calls_log
                ),
                "result": _last_exec_result(tool_calls_log),
                "turns": turn + 1,
                "error": None,
                "tool_calls": tool_calls_log,
                "final_text": clean,
            }

        # Execute the tool call locally
        tool_result = await tool_handler.execute_tool(tc["name"], tc["arguments"])
        tool_response_str = json.dumps(tool_result)

        tool_calls_log.append({
            "name": tc["name"],
            "arguments": tc["arguments"],
            "response": tool_result,
        })

        # Continue the conversation
        prompt = append_tool_turn(prompt, tc, tool_response_str)

    # Exhausted turns
    return {
        "success": False,
        "result": _last_exec_result(tool_calls_log),
        "turns": MAX_TOOL_TURNS,
        "error": "max_turns_exceeded",
        "tool_calls": tool_calls_log,
        "final_text": "",
    }


def _last_exec_result(tool_calls_log: list[dict]) -> Any:
    """Pull the last execute_calc result value from the tool calls log."""
    for tc in reversed(tool_calls_log):
        if tc["name"] == "execute_calc":
            resp = tc.get("response", {})
            outputs = resp.get("outputs")
            if outputs and "result" in outputs:
                return outputs["result"]
    return None


# ── main ─────────────────────────────────────────────────────────────────────

async def async_main():
    parser = argparse.ArgumentParser(description="Local MedCalc-Bench evaluation")
    _root = Path(__file__).resolve().parent.parent
    parser.add_argument("--model", default=str(_root / "outputs" / "medgemma-calc-merged"),
                        help="Path to merged model (default: outputs/medgemma-calc-merged)")
    parser.add_argument("--test-data",
                        default=str(_root / "MedCalc-Bench-Verified" / "datasets" / "test_data.csv"),
                        help="Path to test_data.csv")
    parser.add_argument("--schemas", default="/tmp/all_calc_schemas.json",
                        help="Path to all_calc_schemas.json")
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent / "results" / "eval_local.json"),
                        help="Where to write JSON results")
    parser.add_argument("--calc-filter", default=None,
                        help="Only test a specific calc_id")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max test cases to run")
    args = parser.parse_args()

    # ── Load test data ───────────────────────────────────────────────────────
    print(f"Loading test data from {args.test_data}...")
    test_cases = load_test_data(args.test_data, args.calc_filter)
    if not test_cases:
        print("No test cases found.")
        sys.exit(1)
    if args.limit:
        test_cases = test_cases[:args.limit]
    print(f"  {len(test_cases)} test cases loaded")

    # ── Load schemas + build system prompt ───────────────────────────────────
    with open(args.schemas) as f:
        all_schemas = json.load(f)
    system_prompt = build_system_prompt(all_schemas)

    # ── Load model ───────────────────────────────────────────────────────────
    print(f"Loading model from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto",
    )
    end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    eos_id = tokenizer.eos_token_id
    stop_ids = [end_of_turn_id]
    if eos_id is not None and eos_id != end_of_turn_id:
        stop_ids.append(eos_id)
    print(f"  Model on {model.device}, stop_ids={stop_ids}")

    # ── Init OmniCalc tool handler ───────────────────────────────────────────
    from omnicalc.tools import ToolHandler
    tool_handler = ToolHandler()
    await tool_handler.list_calculators()  # populate cache
    print(f"  {len(tool_handler._calculator_list_cache)} calculators loaded")

    # ── Run ───────────────────────────────────────────────────────────────────
    results = []
    per_calc = defaultdict(lambda: {"total": 0, "correct": 0, "errors": 0})
    total = len(test_cases)

    print(f"\nRunning {total} cases...\n" + "-" * 90)

    for i, tc in enumerate(test_cases, 1):
        calc_id = tc["calc_id"]
        row = tc["row_number"]
        print(f"[{i}/{total}] Row {row}: {tc['calc_name']} ({calc_id})... ", end="", flush=True)

        t0 = time.time()
        tool_handler.reset_session()

        try:
            outcome = await run_one_case(
                model, tokenizer, tool_handler, system_prompt,
                tc["patient_note"], tc["question"], stop_ids,
            )
        except Exception as e:
            outcome = {"success": False, "result": None, "turns": 0,
                       "error": str(e), "tool_calls": [], "final_text": ""}

        elapsed = time.time() - t0
        result_value = outcome["result"]

        chk = check_result(
            result_value, tc["ground_truth"], tc["lower_limit"],
            tc["upper_limit"], tc["output_type"], tc["calc_name"],
        )

        tag = "PASS" if chk["correct"] else ("ERR" if outcome.get("error") else "FAIL")
        got = chk.get("result", result_value)
        print(f"{tag}  got={got}  gt={tc['ground_truth']}  "
              f"turns={outcome['turns']}  [{elapsed:.1f}s]")

        per_calc[tc["calc_name"]]["total"] += 1
        if outcome.get("error"):
            per_calc[tc["calc_name"]]["errors"] += 1
        elif chk["correct"]:
            per_calc[tc["calc_name"]]["correct"] += 1

        results.append({
            **tc,
            "correct": chk["correct"],
            "result_value": chk.get("result"),
            "turns": outcome["turns"],
            "error": outcome.get("error"),
            "final_text": outcome.get("final_text", ""),
            "elapsed_s": round(elapsed, 2),
            "tool_calls": [
                {"name": t["name"], "arguments": t["arguments"]}
                for t in outcome.get("tool_calls", [])
            ],
            "check_details": chk,
        })

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("RESULTS SUMMARY")
    print("=" * 90)
    print(f"\n{'Calculator':<55} {'Acc':>6} {'Correct':>8} {'Total':>6} {'Err':>4}")
    print("-" * 90)

    total_correct = sum(1 for r in results if r["correct"])
    total_errors = sum(1 for r in results if r.get("error"))
    summary_rows = []

    for name in sorted(per_calc):
        s = per_calc[name]
        n = s["total"] - s["errors"]
        acc = s["correct"] / n * 100 if n > 0 else 0
        print(f"{name:<55} {acc:>5.1f}% {s['correct']:>8}/{n:<6} {s['errors']:>3}E")
        summary_rows.append({"calculator": name, "accuracy": round(acc, 2),
                             "correct": s["correct"], "total": n, "errors": s["errors"]})

    ne = total - total_errors
    oa = total_correct / ne * 100 if ne > 0 else 0
    print("-" * 90)
    print(f"{'OVERALL':<55} {oa:>5.1f}% {total_correct:>8}/{ne:<6} {total_errors:>3}E")
    print("=" * 90)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model": args.model,
                "test_data": args.test_data,
                "calc_filter": args.calc_filter,
                "total_cases": total,
            },
            "summary": {
                "overall_accuracy": round(oa, 2),
                "total_correct": total_correct,
                "total_evaluated": ne,
                "total_errors": total_errors,
                "per_calculator": summary_rows,
            },
            "results": results,
        }, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
