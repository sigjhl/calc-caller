"""
Calculator schemas: defines calc_id, input field IDs, units, and the mapping
back to MedCalc function parameter names used by the executor.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MEDCALC_DIR = Path(__file__).parent.parent / "MedCalc-Bench-Verified"
_NAME_TO_PYTHON_PATH = MEDCALC_DIR / "calculator_implementations" / "name_to_python.json"

# ── calc_id assignment ────────────────────────────────────────────────────────
MEDCALC_ID_TO_CALC_ID: dict[int, str] = {
    2:  "creatinine_clearance",
    3:  "ckd_epi_gfr",
    4:  "cha2ds2_vasc",
    5:  "mean_arterial_pressure",
    6:  "bmi",
    7:  "calcium_correction",
    8:  "wells_pe",
    9:  "mdrd_gfr",
    10: "ideal_body_weight",
    11: "qtc_bazett",
    13: "estimated_due_date",
    15: "child_pugh",
    16: "wells_dvt",
    17: "rcri",
    18: "heart_score",
    19: "fib4",
    20: "centor_score",
    21: "glasgow_coma_score",
    22: "maintenance_fluids",
    23: "meld_na",
    24: "steroid_conversion",
    25: "has_bled",
    26: "sodium_correction_hyperglycemia",
    27: "glasgow_blatchford",
    28: "apache2",
    29: "psi_cap",
    30: "serum_osmolality",
    31: "homa_ir",
    32: "cci",
    33: "feverpain",
    36: "caprini_vte",
    38: "free_water_deficit",
    39: "anion_gap",
    40: "fena",
    43: "sofa",
    44: "ldl_calculated",
    45: "curb65",
    46: "framingham_chd",
    48: "perc_rule",
    49: "mme",
    51: "sirs",
    56: "qtc_fridericia",
    57: "qtc_framingham",
    58: "qtc_hodges",
    59: "qtc_rautaharju",
    60: "bsa",
    61: "target_weight",
    62: "adjusted_body_weight",
    63: "delta_gap",
    64: "delta_ratio",
    65: "albumin_corrected_anion_gap",
    66: "albumin_corrected_delta_gap",
    67: "albumin_corrected_delta_ratio",
    68: "estimated_conception_date",
    69: "estimated_gestational_age",
}

CALC_ID_TO_MEDCALC_ID: dict[str, int] = {v: k for k, v in MEDCALC_ID_TO_CALC_ID.items()}

_TYPE_TO_TAGS: dict[str, list[str]] = {
    "lab test":  ["laboratory"],
    "physical":  ["physical"],
    "risk":      ["risk"],
    "severity":  ["severity"],
    "diagnosis": ["diagnosis"],
    "dosage":    ["dosage"],
    "date":      ["obstetrics"],
}

# ── field definitions ─────────────────────────────────────────────────────────
# note_field_name → (field_id, canonical_unit, field_type, synonyms, medcalc_fmt)
# medcalc_fmt: "list" = [value, unit], "scalar" = bare value, "boolean" = bool
_FIELD_DEFS: dict[str, tuple[str, str, str, list[str], str]] = {
    # demographics
    "age":    ("age",    "years", "number",      ["age"],              "list"),
    "sex":    ("sex",    "",      "categorical",  ["sex", "gender"],    "scalar"),
    "Race":   ("race",   "",      "categorical",  ["race", "ethnicity"],"scalar"),
    # physical
    "weight": ("weight", "kg",   "number", ["weight", "wt"],           "list"),
    "height": ("height", "cm",   "number", ["height", "ht"],           "list"),
    "Body Mass Index (BMI)": ("bmi", "kg/m²", "number",
                              ["bmi", "body mass index"],              "list"),
    # serum chemistry
    "creatinine":   ("serum_creatinine", "mg/dL", "number",
                     ["cr", "creatinine", "creat"],                    "list"),
    "Calcium":      ("serum_calcium",    "mg/dL", "number",
                     ["ca", "calcium"],                                "list"),
    "Albumin":      ("serum_albumin",    "g/dL",  "number",
                     ["albumin", "alb"],                               "list"),
    "Bilirubin":    ("serum_bilirubin",  "mg/dL", "number",
                     ["bili", "total bilirubin"],                      "list"),
    "Sodium":       ("serum_sodium",     "mEq/L", "number",
                     ["na", "sodium"],                                 "list"),
    "Glucose":      ("serum_glucose",    "mg/dL", "number",
                     ["glucose", "blood glucose"],                     "list"),
    "Potassium":    ("serum_potassium",  "mEq/L", "number",
                     ["k", "potassium"],                               "list"),
    "Chloride":     ("serum_chloride",   "mEq/L", "number",
                     ["cl", "chloride"],                               "list"),
    "Bicarbonate":  ("serum_bicarbonate","mEq/L", "number",
                     ["hco3", "bicarb"],                               "list"),
    "Insulin":      ("serum_insulin",    "μIU/mL","number",
                     ["insulin"],                                      "list"),
    "Hemoglobin":   ("hemoglobin",       "g/dL",  "number",
                     ["hgb", "hb"],                                    "list"),
    "Hematocrit":   ("hematocrit",       "%",     "number",
                     ["hct"],                                          "list"),
    "White blood cell count": ("wbc", "10^9/L", "number",
                     ["wbc", "leukocytes"],                            "list"),
    "Platelet count": ("platelet_count", "10^9/L", "number",
                     ["platelets", "plt"],                             "list"),
    "Alanine aminotransferase":  ("alt", "IU/L", "number",
                     ["alt", "sgpt"],                                  "list"),
    "Aspartate aminotransferase":("ast", "IU/L", "number",
                     ["ast", "sgot"],                                  "list"),
    "Blood Urea Nitrogen (BUN)": ("bun", "mg/dL", "number",
                     ["bun"],                                          "list"),
    "Triglycerides": ("triglycerides",    "mg/dL", "number",
                     ["tg", "triglycerides"],                          "list"),
    "Total cholesterol": ("total_cholesterol", "mg/dL", "number",
                     ["cholesterol"],                                  "list"),
    "high-density lipoprotein cholesterol": ("hdl_cholesterol", "mg/dL", "number",
                     ["hdl"],                                          "list"),
    # coagulation
    "international normalized ratio": ("inr", "", "number",
                     ["inr", "prothrombin time ratio"],                "scalar"),
    "Labile international normalized ratio": ("labile_inr", "", "boolean",
                     ["labile inr"],                                   "boolean"),
    # vitals
    "Heart Rate or Pulse":   ("heart_rate",     "beats/min", "number",
                     ["heart rate", "pulse", "hr"],                    "list"),
    "Systolic Blood Pressure":  ("systolic_bp",  "mmHg", "number",
                     ["sbp", "systolic blood pressure"],               "list"),
    "Diastolic Blood Pressure": ("diastolic_bp", "mmHg", "number",
                     ["dbp", "diastolic blood pressure"],              "list"),
    "Temperature":   ("temperature",      "°C",  "number",
                     ["temp", "temperature"],                          "list"),
    "respiratory rate": ("respiratory_rate", "breaths/min", "number",
                     ["rr", "resp rate"],                              "list"),
    "Mean arterial pressure": ("mean_arterial_pressure", "mmHg", "number",
                     ["map", "mean arterial pressure"],                "list"),
    # respiratory / gases
    "FiO2":  ("fio2",  "%",    "number", ["fio2"],                    "list"),
    "PaO2":  ("pao2",  "mmHg", "number", ["pao2"],                    "list"),
    "PaCO2": ("paco2", "mmHg", "number", ["paco2"],                   "list"),
    "pH":    ("ph",    "",     "number", ["ph"],                      "scalar"),
    "O₂ saturation percentage": ("o2_saturation", "%", "number",
                     ["o2 sat", "spo2"],                               "list"),
    "Partial pressure of oxygen": ("partial_pressure_o2", "mmHg", "number",
                     ["po2"],                                          "list"),
    "A-a gradient": ("aa_gradient", "mmHg", "number",
                     ["a-a gradient"],                                 "list"),
    # urine
    "Urine creatinine": ("urine_creatinine", "mg/dL",  "number",
                     ["urine creatinine"],                             "list"),
    "Urine sodium":     ("urine_sodium",     "mEq/L",  "number",
                     ["urine sodium"],                                 "list"),
    "Urine Output":     ("urine_output",     "mL/day", "number",
                     ["urine output"],                                 "list"),
    # ECG
    "QT Interval":  ("qt_interval", "msec", "number", ["qt"],         "list"),
    "QT interval":  ("qt_interval", "msec", "number", ["qt"],         "list"),
    # GCS components
    "Best motor response":  ("gcs_motor",  "", "integer",
                     ["motor response"],                               "scalar"),
    "Best eye response":    ("gcs_eye",    "", "integer",
                     ["eye response"],                                 "scalar"),
    "Best verbal response": ("gcs_verbal", "", "integer",
                     ["verbal response"],                              "scalar"),
    "Glasgow Coma Score":   ("glasgow_coma_score", "", "number",
                     ["gcs"],                                          "scalar"),
    # obstetric dates
    "Last menstrual date": ("last_menstrual_date", "", "date",
                     ["lmp", "last menstrual period"],                 "scalar"),
    "Current Date":        ("current_date",        "", "date",
                     ["current date", "today"],                        "scalar"),
    "cycle length":        ("cycle_length", "days", "number",
                     ["cycle length"],                                 "scalar"),
    # dialysis flags (MELD-Na)
    "Dialysis at least twice in the past week": (
        "dialysis_twice", "", "boolean", ["dialysis twice weekly"],    "boolean"),
    "Continuous veno-venous hemodialysis for ≥24 hours in the past week": (
        "cvvhd", "", "boolean", ["cvvhd"],                            "boolean"),
    # SOFA vasopressors / devices
    "On mechanical ventilation":       ("on_mechanical_ventilation", "", "boolean",
                     ["mechanical ventilation"],                       "boolean"),
    "Continuous positive airway pressure": ("cpap", "", "boolean",
                     ["cpap"],                                         "boolean"),
    "Hypotension":     ("hypotension",   "", "boolean", ["hypotension"],   "boolean"),
    "DOPamine":        ("dopamine",    "mcg/kg/min", "number",
                     ["dopamine"],                                     "list"),
    "DOBUTamine":      ("dobutamine",  "mcg/kg/min", "number",
                     ["dobutamine"],                                   "list"),
    "norEPINEPHrine":  ("norepinephrine", "mcg/kg/min", "number",
                     ["norepinephrine", "norepi"],                     "list"),
    "EPINEPHrine":     ("epinephrine",   "mcg/kg/min", "number",
                     ["epinephrine", "epi"],                           "list"),
    # steroid conversion
    "input steroid":  ("input_steroid",  "", "text", ["current steroid"],  "scalar"),
    "target steroid": ("target_steroid", "", "text", ["desired steroid"],  "scalar"),
    # ── HEART Score: categorical fields ───────────────────────────────────────
    "Suspicion History":     ("history",          "", "categorical",
                     ["suspicion", "history"],                         "scalar"),
    "Electrocardiogram Test":("electrocardiogram","", "categorical",
                     ["ekg", "ecg", "electrocardiogram"],             "scalar"),
    "Initial troponin":      ("initial_troponin", "", "categorical",
                     ["troponin", "initial troponin"],                 "scalar"),
    # ── Caprini VTE: categorical fields ───────────────────────────────────────
    "Surgery Type":  ("surgery_type", "", "categorical",
                     ["surgery type"],                                 "scalar"),
    "Mobility":      ("mobility",     "", "categorical",
                     ["mobility"],                                     "scalar"),
    # ── RCRI: pre-op creatinine is a numeric lab value, not a boolean ─────────
    "Pre-operative creatinine": ("pre_operative_creatinine", "mg/dL", "number",
                     ["pre-op creatinine"],                            "list"),
    # ── Child-Pugh categorical fields ───────────────────────────────────────
    "Ascites":                ("ascites",           "", "categorical", ["ascites"],       "scalar"),
    "Encephalopathy":         ("encephalopathy",    "", "categorical", ["encephalopathy"],"scalar"),
    # ── CCI categorical fields ──────────────────────────────────────────────
    "Liver disease severity": ("liver_disease",     "", "categorical", ["liver disease"], "scalar"),
    "Diabetes mellitus":      ("diabetes_mellitus", "", "categorical", ["diabetes"],      "scalar"),
    "Solid tumor":            ("solid_tumor",       "", "categorical", ["solid tumor"],   "scalar"),
}

# Metadata-only keys to skip when iterating name_to_python.json parameters
_SKIP_KEYS = {"file path", "explanation function", "calculator name", "type",
              "question", "Question"}


def _infer_fallback(func_param: str) -> tuple[str, str, str, list[str], str]:
    """For clinical flag params not in _FIELD_DEFS, treat as boolean."""
    return func_param, "", "boolean", [], "boolean"


# ── MME: supported drugs & conversion factors ────────────────────────────────
MME_DRUGS: dict[str, float] = {
    "Codeine":          0.15,
    "FentaNYL buccal":  0.13,
    "FentaNYL patch":   2.4,
    "HYDROcodone":      1,
    "HYDROmorphone":    5,
    "Methadone":        4.7,
    "Morphine":         1,
    "OxyCODONE":        1.5,
    "OxyMORphone":      3,
    "Tapentadol":       0.4,
    "TraMADol":         0.2,
    "Buprenorphine":    10,
}


def _build_mme_schema() -> dict[str, Any]:
    """Hand-crafted schema for the MME calculator (dynamic drug inputs)."""
    return {
        "calc_id":     "mme",
        "medcalc_id":  49,
        "title":       "Morphine Milligram Equivalents (MME) Calculator",
        "description": (
            "Calculates the patient's daily Morphine Milligram Equivalents "
            "(MME) from their opioid prescriptions using the CDC 2022 "
            "conversion factors."
        ),
        "version":     "1.0",
        "tags":        ["dosage"],
        "file_path":   "mme.py",
        "function":    "mme_explanation",
        "inputs": [{
            "id":             "medications",
            "label":          "Opioid medications",
            "type":           "array",
            "required":       True,
            "canonical_unit": "",
            "synonyms":       ["opioids", "narcotics"],
            "constraints": {
                "item_schema": {
                    "drug":          "One of: " + ", ".join(MME_DRUGS.keys()),
                    "dose":          "Numeric dose per administration",
                    "dose_unit":     "mg for most opioids; mcg for fentaNYL",
                    "doses_per_day": "Number of doses taken per day",
                },
                "mme_conversion_factors": MME_DRUGS,
            },
            # executor-only:
            "function_param": "medications",
            "medcalc_fmt":    "mme_special",
        }],
        "presets": [],
    }


def load_schemas() -> dict[str, dict[str, Any]]:
    """
    Return a dict keyed by calc_id.  Each value is the full schema including
    internal fields (function_param, medcalc_fmt) used by the executor.
    """
    with open(_NAME_TO_PYTHON_PATH) as f:
        n2p: dict[str, dict] = json.load(f)

    schemas: dict[str, dict] = {}

    for medcalc_id_str, entry in n2p.items():
        medcalc_id = int(medcalc_id_str)
        calc_id = MEDCALC_ID_TO_CALC_ID.get(medcalc_id)
        if not calc_id:
            continue

        # MME uses a hand-crafted schema (dynamic drug inputs)
        if calc_id == "mme":
            schemas["mme"] = _build_mme_schema()
            continue

        params = {k: v for k, v in entry.items() if k not in _SKIP_KEYS}

        inputs: list[dict] = []
        for note_field, func_param in params.items():
            if note_field in _FIELD_DEFS:
                fid, unit, ftype, syns, fmt = _FIELD_DEFS[note_field]
            else:
                fid, unit, ftype, syns, fmt = _infer_fallback(func_param)

            inputs.append({
                "id":             fid,
                "label":          note_field,
                "type":           ftype,
                "required":       True,
                "canonical_unit": unit,
                "synonyms":       syns,
                "constraints":    {},
                # executor-only fields:
                "function_param": func_param,
                "medcalc_fmt":    fmt,
            })

        tags = _TYPE_TO_TAGS.get(entry.get("type", ""), [])
        schemas[calc_id] = {
            "calc_id":     calc_id,
            "medcalc_id":  medcalc_id,
            "title":       entry.get("calculator name", calc_id),
            "description": entry.get("question", ""),
            "version":     "1.0",
            "tags":        tags,
            "file_path":   entry.get("file path", ""),
            "function":    entry.get("explanation function", ""),
            "inputs":      inputs,
            "presets":     [],
        }

    return schemas


def public_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Strip executor-only fields before sending to the model."""
    pub_inputs = [
        {k: v for k, v in inp.items() if k not in ("function_param", "medcalc_fmt")}
        for inp in schema["inputs"]
    ]
    return {
        "calc_id":     schema["calc_id"],
        "title":       schema["title"],
        "description": schema["description"],
        "version":     schema["version"],
        "tags":        schema["tags"],
        "inputs":      pub_inputs,
        "presets":     schema["presets"],
    }
