# MedGemma Calculator Tool-Use Fine-Tuning

Fine-tuning [MedGemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it) with QLoRA to perform clinical calculator tool-use on the [MedCalc-Bench](https://arxiv.org/abs/2406.12036) benchmark.

## Approach

Rather than asking the LLM to recall formulas and perform arithmetic, we delegate all computation to a deterministic calculator backend (**OmniCalc**) and train the model exclusively on the extraction-and-calling task. Each training example is a complete multi-turn tool-use trajectory:

```
system prompt → user (clinical note)
→ model calls calc_info → tool returns schema
→ model calls execute_calc with extracted values → tool returns result
→ model responds "Done."
```

This eliminates formula recall and arithmetic errors entirely — the model only needs to identify the right calculator and extract the right values.

## Repository Structure

```
calc_caller/
├── train.py                    # QLoRA fine-tuning script (Unsloth + TRL)
├── src/
│   ├── prompts.py              # System prompt & tool schema definitions
│   ├── schemas.py              # MedCalc-Bench → OmniCalc field mappings
│   └── executor.py             # Calculator execution wrapper
├── scripts/
│   └── build_training_data.py  # Converts MedCalc-Bench CSV → training JSONL
├── eval/
│   ├── eval_local.py           # Local evaluation (no server needed)
│   ├── omnicalc/               # Calculator backends with unit conversion
│   └── results/                # Evaluation results
├── MODEL_CARD.md               # HuggingFace model card
└── data/                       # Generated training data (gitignored)
```

## Setup

```bash
# Install dependencies
uv sync
uv pip install "unsloth[cu128-torch2100]"  # adjust for your CUDA/torch version
```

### Download the dataset

The training and evaluation scripts expect the [MedCalc-Bench-Verified](https://github.com/nikhilk7153/MedCalc-Bench-Verified) dataset to be cloned into the repo root. This dataset is released under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) and is not included in this repository.

```bash
git clone https://github.com/nikhilk7153/MedCalc-Bench-Verified.git
```

After cloning, the expected layout is:

```
calc_caller/
└── MedCalc-Bench-Verified/
    └── datasets/
        ├── train_data.csv   # used by build_training_data.py
        └── test_data.csv    # used by eval/eval_local.py
```

## Training

### 1. Build training data

Converts MedCalc-Bench `train_data.csv` into tool-use trajectories:

```bash
uv run python scripts/build_training_data.py --out data/train.jsonl
```

### 2. Train

```bash
uv run python train.py --epochs 2 --output outputs/medgemma-calc
```

### Hyperparameters

| Parameter | Value |
|---|---|
| Method | QLoRA (4-bit NF4) |
| LoRA rank / alpha | 16 / 16 |
| rsLoRA | true |
| Target modules | all linear layers |
| Optimizer | AdamW 8-bit |
| Learning rate | 2e-4 (cosine, 5% warmup) |
| Weight decay | 0.01 |
| Effective batch size | 16 (2 × 8 grad accum) |
| Max sequence length | 5,376 tokens |
| Epochs | 2 |
| Precision | bfloat16 |

### Data

- **Source**: MedCalc-Bench `train_data.csv` — 10,496 examples after trajectory transformation
- **Format**: OpenAI-style multi-turn messages converted to Gemma 3 chat template with [LM Studio](https://lmstudio.ai/)-compatible `[TOOL_REQUEST]...[END_TOOL_REQUEST]` tool-call markers
- **Masking**: Only model turns are supervised; system, user, and tool turns are masked

## Evaluation

`eval/eval_local.py` runs the merged model locally through the same multi-turn tool-use loop against OmniCalc calculator backends with greedy decoding. No external server needed.

```bash
uv run python eval/eval_local.py --model outputs/medgemma-calc-merged
```

### Results: 84.6% (931/1,100)

55 calculators, 20 test cases each.

| Accuracy | Count | Calculators |
|---|---|---|
| **100%** | 25 | Adjusted Body Weight, Albumin-Corrected AG, Albumin-Corrected Delta Gap, Anion Gap, BMI, BSA, CrCl, Delta Gap, Delta Ratio, Est. Conception Date, Est. Due Date, Est. Gestational Age, HOMA-IR, IBW, Maintenance Fluids, MAP, MDRD GFR, MME, QTc (×5), Serum Osmolality, Sodium Correction, Target Weight |
| **90–99%** | 8 | CKD-EPI, FIB-4, Free Water Deficit, LDL, Albumin-Corrected Delta Ratio, CURB-65, FENa, MELD Na |
| **75–89%** | 8 | Steroid Conversion, Calcium Correction, Child-Pugh, PERC Rule, RCRI, Wells DVT, Framingham Risk, GCS |
| **50–74%** | 8 | CHA₂DS₂-VASc, SIRS, Glasgow-Blatchford, HAS-BLED, PSI/PORT, SOFA, Centor, CCI, FeverPAIN |
| **< 50%** | 6 | APACHE II, Caprini, Wells PE, HEART |

Most remaining errors are clinical reading comprehension failures on complex multi-criteria scoring systems, not arithmetic errors.

## License

- **Code**: Apache 2.0
- **Model weights**: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) (inherited from MedCalc-Bench)
- **Base model**: Subject to [HAI-DEF Terms of Use](https://developers.google.com/health-ai-developer-foundations/terms)

## Citation

```bibtex
@article{khandekar2024medcalcbench,
  title={MedCalc-Bench: Evaluating Large Language Models for Medical Calculations},
  author={Khandekar, Nikhil and Dey, Sestina and Matero, Matthew and Shrestha, Amanuel and Mahowald, Kyle and Ungar, Lyle},
  journal={arXiv preprint arXiv:2406.12036},
  year={2024}
}
```
