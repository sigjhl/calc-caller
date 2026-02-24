---
license: cc-by-sa-4.0
base_model: google/medgemma-1.5-4b-it
library_name: transformers
pipeline_tag: text-generation
tags:
- medical
- clinical-calculation
- tool-use
- qlora
- medcalc-bench
datasets:
- MedCalc-Bench
---

# MedGemma-1.5-4B-It — MedCalcCaller

A fine-tuned version of [google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it) specialized for **clinical calculator tool-use**.
Instead of performing arithmetic, the model extracts clinical parameters from unstructured text and generates structured tool calls to a deterministic calculator backend.

## Approach

Standard LLMs fail at medical calculations due to three compounding error sources: entity extraction, formula recall, and arithmetic.
This model eliminates the latter two entirely by delegating computation to a symbolic calculator engine (**OmniCalc**) and training the LLM exclusively on the extraction-and-calling task.

Each training example is a complete multi-turn tool-use trajectory:

```
system prompt → user (clinical note) → model calls calc_info → tool returns schema
→ model calls execute_calc with extracted values → tool returns result → model responds "Done."
```

The model learns to:
1. Identify the correct calculator from the clinical context
2. Call `calc_info` to retrieve the exact input field schema
3. Extract values (with units when needed) from the clinical note
4. Call `execute_calc` with properly structured arguments

## Supported Calculators

55 clinical calculators covering formulae, risk scores, and date calculations:

<details>
<summary>Full list (click to expand)</summary>

| Category | Calculators |
|---|---|
| **Formulae** | Adjusted Body Weight, Anion Gap, Albumin-Corrected Anion Gap, Albumin-Corrected Delta Gap, Albumin-Corrected Delta Ratio, BMI, Body Surface Area, Calcium Correction for Hypoalbuminemia, CKD-EPI GFR, Creatinine Clearance (Cockcroft-Gault), Delta Gap, Delta Ratio, FENa, FIB-4 Index, Free Water Deficit, HOMA-IR, Ideal Body Weight, LDL Calculated, Maintenance Fluids, MAP, MDRD GFR, MELD Na (UNOS/OPTN), MME Calculator, QTc (Bazett, Framingham, Fridericia, Hodges, Rautaharju), Serum Osmolality, Sodium Correction for Hyperglycemia, Steroid Conversion, Target Weight |
| **Risk Scores** | APACHE II, Caprini VTE, CHA₂DS₂-VASc, Centor (Modified/McIsaac), Charlson Comorbidity Index, Child-Pugh, CURB-65, FeverPAIN, Framingham Risk Score, GCS, Glasgow-Blatchford Bleeding Score, HAS-BLED, HEART Score, PERC Rule, PSI/PORT Score, Revised Cardiac Risk Index, SIRS Criteria, SOFA Score, Wells' Criteria (DVT), Wells' Criteria (PE) |
| **Date Calculations** | Estimated Gestational Age, Estimated Due Date, Estimated Date of Conception |

</details>

## Training Details

### Data

- **Source**: [MedCalc-Bench](https://arxiv.org/abs/2406.12036) `train_data.csv` — 10,496 examples after trajectory transformation (42 skipped due to missing/ambiguous fields)
- **Transformation**: Each static (note, answer) pair was converted into a multi-turn tool-use trajectory using the OmniCalc calculator backend. The model sees the tool schema, extracts values, and receives the execution result.
- **Tool format**: [LM Studio](https://lmstudio.ai/)-compatible `[TOOL_REQUEST]...[END_TOOL_REQUEST]` markers within the Gemma 3 chat template (`<start_of_turn>`/`<end_of_turn>`)

### Hyperparameters

| Parameter | Value |
|---|---|
| Method | QLoRA (4-bit NF4 quantized base) |
| LoRA rank (r) | 16 |
| LoRA alpha | 16 |
| rsLoRA | true |
| Target modules | q, k, v, o, gate, up, down projections |
| Trainable parameters | 29.8M / 4.33B (0.69%) |
| Optimizer | AdamW 8-bit |
| Learning rate | 2e-4 (cosine schedule, 5% warmup) |
| Weight decay | 0.01 |
| Effective batch size | 16 (2 per device × 8 gradient accumulation) |
| Max sequence length | 5,376 tokens |
| Epochs | 2 |
| Total steps | 1,312 |
| Final training loss | 0.12 |
| Precision | bfloat16 |
| Hardware | 1× NVIDIA RTX A6000 (48 GB) |
| Framework | Unsloth + TRL SFTTrainer |
| Response masking | Only model turns are supervised (system/user/tool turns masked with -100) |

### Training Notes

- Vision tower and multi-modal projector are frozen (text-only fine-tuning)
- Unsloth's SDPA attention patch was replaced with the original transformers FA2 code path to fix a mask-shape bug with padded batches
- No packing — each example is a separate sequence

## Evaluation

Evaluated on the full [MedCalc-Bench](https://arxiv.org/abs/2406.12036) test set (1,100 instances, 20 per calculator) using greedy decoding through the same multi-turn tool-use loop against the OmniCalc calculator backend.

### Overall: 84.6% (931/1,100)

| Accuracy | Count | Calculators |
|---|---|---|
| **100%** | 25 | Adjusted Body Weight, Albumin-Corrected Anion Gap, Albumin-Corrected Delta Gap, Anion Gap, BMI, Body Surface Area, Creatinine Clearance, Delta Gap, Delta Ratio, Est. Conception Date, Est. Due Date, Est. Gestational Age, HOMA-IR, Ideal Body Weight, Maintenance Fluids, MAP, MDRD GFR, MME, QTc Bazett, QTc Framingham, QTc Fridericia, QTc Hodges, QTc Rautaharju, Serum Osmolality, Sodium Correction, Target Weight |
| **90–99%** | 8 | CKD-EPI 95%, FIB-4 95%, Free Water Deficit 95%, LDL 95%, Albumin-Corrected Delta Ratio 90%, CURB-65 90%, FENa 90%, MELD Na 90% |
| **75–89%** | 8 | Steroid Conversion 85%, Calcium Correction 80%, Child-Pugh 80%, PERC Rule 80%, RCRI 80%, Wells DVT 80%, Framingham Risk 75%, GCS 75% |
| **50–74%** | 8 | CHA₂DS₂-VASc 70%, SIRS 70%, Glasgow-Blatchford 60%, HAS-BLED 60%, PSI/PORT 65%, SOFA 65%, Centor 50%, CCI 50%, FeverPAIN 50% |
| **< 50%** | 6 | APACHE II 40%, Caprini 40%, Wells PE 40%, HEART 20% |

### Error Analysis

Calculators at 100% are primarily formula-based (the model only needs to extract 2–5 numeric values). Most remaining errors are **clinical reading comprehension failures** on complex scoring systems — the model misreads or omits criteria from lengthy clinical notes, not arithmetic errors.

## Usage

### With the OmniCalc backend (recommended)

This model is designed to be used with the OmniCalc calculator backend included in the [training repository](https://github.com/YOUR_USERNAME/calc-caller). The backend handles all computation, unit conversion, and validation.

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/calc-caller
cd calc-caller

# Run evaluation
uv run python eval/eval_local.py --model path/to/this/model
```

### With LM Studio

Load the model in [LM Studio](https://lmstudio.ai/) and configure the OmniCalc tools. The model's tool-call format (`[TOOL_REQUEST]...[END_TOOL_REQUEST]`) matches LM Studio's default tool-calling convention for models without native tool support.

### Prompt Format

The model expects the Gemma 3 chat template with tool definitions injected into the system prompt:

```
<bos><start_of_turn>system
You are OmniCalc, a clinical calculator assistant.
[... system prompt with calculator list and rules ...]

You can request calls to available tools with this EXACT format:
[TOOL_REQUEST]{"name": "tool_name", "arguments": {"param1": "value1"}}[END_TOOL_REQUEST]

AVAILABLE TOOLS:
[... tool schemas for calc_info and execute_calc ...]
<end_of_turn>
<start_of_turn>user
[clinical note with calculation request]<end_of_turn>
<start_of_turn>model
[TOOL_REQUEST]{"name": "calc_info", "arguments": {"calc_id": "..."}}[END_TOOL_REQUEST]<end_of_turn>
<start_of_turn>user
[tool response with schema]<end_of_turn>
<start_of_turn>model
[TOOL_REQUEST]{"name": "execute_calc", "arguments": {"calc_id": "...", "variables": {...}}}[END_TOOL_REQUEST]<end_of_turn>
```

## Limitations

- **Not for clinical use.** This model is a research prototype. It must not be used for real-world patient diagnosis or treatment.
- **Extraction errors on complex scores.** Scoring systems with many binary criteria (HEART, Caprini, Wells PE) remain challenging — the model may miss or misinterpret criteria from lengthy clinical notes.
- **Requires a calculator backend.** The model does not perform arithmetic. It must be paired with a compatible calculator engine to produce results.
- **English only.** Trained exclusively on English clinical notes.

## Licensing & Terms

This model is a **Model Derivative** as defined in the [Health AI Developer Foundations (HAI-DEF) Terms of Use](https://developers.google.com/health-ai-developer-foundations/terms).

- **Foundational model**: Use is subject to the [HAI-DEF Terms of Use](https://developers.google.com/health-ai-developer-foundations/terms) and the [HAI-DEF Prohibited Use Policy](https://developers.google.com/health-ai-developer-foundations/prohibited-use-policy).
- **Training data**: These weights are released under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) per the requirements of the MedCalc-Bench dataset. Any further fine-tuning or redistribution must use the same license.
- **Medical disclaimer**: This model is not a medical device and is not intended for clinical use. It must not be used in any way that would cause a Health Regulatory Authority to deem Google to be a "manufacturer" of a medical device.

## Citation

If you use this model, please cite:

```bibtex
@article{khandekar2024medcalcbench,
  title={MedCalc-Bench: Evaluating Large Language Models for Medical Calculations},
  author={Khandekar, Nikhil and Dey, Sestina and Matero, Matthew and Shrestha, Amanuel and Mahowald, Kyle and Ungar, Lyle},
  journal={arXiv preprint arXiv:2406.12036},
  year={2024}
}
```

## NOTICE

HAI-DEF is provided under and subject to the Health AI Developer Foundations Terms of Use found at https://developers.google.com/health-ai-developer-foundations/terms
