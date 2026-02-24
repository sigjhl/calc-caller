"""
QLoRA fine-tuning of MedGemma on clinical calculator tool-use trajectories.

Requirements (install before running):
    uv sync
    uv pip install "unsloth[cu128-torch2100]"   # adjust cu/torch version to match your system

Run:
    uv run train.py [--data data/train.jsonl] [--output outputs/medgemma-calc]
"""
from __future__ import annotations

import argparse
import json
import os
import random

# Disable Unsloth's flex_attention and torch.compile patches so that
# transformers' native FA2 code path is used instead.
os.environ["UNSLOTH_ENABLE_FLEX_ATTENTION"] = "0"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

# Save the original Gemma3 attention forward BEFORE Unsloth monkey-patches it.
# Unsloth's patched SDPA has a mask-shape bug with padded batches.
import transformers.models.gemma3.modeling_gemma3 as _g3_module
_ORIGINAL_GEMMA3_ATTN_FORWARD = _g3_module.Gemma3Attention.forward

import torch
import wandb
from datasets import load_dataset, Dataset
from transformers import TrainerCallback, Trainer as _HFTrainer
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

from src.prompts import TOOL_SCHEMAS

# ── defaults ──────────────────────────────────────────────────────────────────
MODEL_NAME     = "google/medgemma-1.5-4b-it"
MAX_SEQ_LEN    = 5376          # covers longest sample at 5352 tokens
LORA_RANK      = 16
LORA_ALPHA     = 16
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# LM Studio's default tool-calling instructions (injected into system prompt
# for models without native tool support like Gemma).
_TOOL_INSTRUCTIONS = (
    "You can request calls to available tools with this EXACT format:\n"
    '[TOOL_REQUEST]{{"name": "tool_name", "arguments": {{"param1": "value1"}}}}[END_TOOL_REQUEST]\n\n'
    "AVAILABLE TOOLS:\n"
    "{tools_json}\n\n"
    "RULES:\n"
    "- Only use tools from AVAILABLE TOOLS\n"
    "- Include all required arguments\n"
    "- Use one [TOOL_REQUEST] block per tool\n"
    "- Never use [TOOL_RESULT]\n"
    "- If you decide to call one or more tools, there should be no other text in your message"
)

N_VAL = 0
N_GENERATE_SAMPLES = 3


def _build_tool_block() -> str:
    """Build the AVAILABLE TOOLS block matching LM Studio's default format."""
    tools_json = json.dumps({"type": "toolArray", "tools": TOOL_SCHEMAS}, indent=2)
    return _TOOL_INSTRUCTIONS.format(tools_json=tools_json)


def format_to_gemma(messages: list[dict]) -> str:
    """Convert OpenAI-format messages to Gemma tokens with LM Studio default tool format."""
    tool_block = _build_tool_block()
    parts = ["<bos>"]

    for msg in messages:
        role = msg["role"]

        if role == "system":
            parts.append(
                f"<start_of_turn>system\n{msg['content']}\n\n{tool_block}<end_of_turn>\n"
            )
        elif role == "user":
            parts.append(f"<start_of_turn>user\n{msg['content']}<end_of_turn>\n")
        elif role == "assistant":
            if msg.get("tool_calls"):
                tc_texts = []
                for tc in msg["tool_calls"]:
                    name = tc["function"]["name"]
                    arguments = tc["function"]["arguments"]
                    tc_texts.append(
                        f'[TOOL_REQUEST]{{"name": "{name}", "arguments": {arguments}}}[END_TOOL_REQUEST]'
                    )
                parts.append(
                    f"<start_of_turn>model\n{''.join(tc_texts)}<end_of_turn>\n"
                )
            else:
                parts.append(
                    f"<start_of_turn>model\n{msg['content']}<end_of_turn>\n"
                )
        elif role == "tool":
            parts.append(f"<start_of_turn>user\n{msg['content']}<end_of_turn>\n")

    return "".join(parts)


def make_prompt_prefix(messages: list[dict]) -> str:
    """Build the prompt up to (but not including) the first model response."""
    tool_block = _build_tool_block()
    parts = ["<bos>"]

    for msg in messages:
        role = msg["role"]
        if role == "system":
            parts.append(
                f"<start_of_turn>system\n{msg['content']}\n\n{tool_block}<end_of_turn>\n"
            )
        elif role == "user":
            parts.append(f"<start_of_turn>user\n{msg['content']}<end_of_turn>\n")
        elif role == "assistant":
            break

    parts.append("<start_of_turn>model\n")
    return "".join(parts)


# ── Pre-tokenization (avoids Gemma3 tokenizer pickle issues) ─────────────────

def _find_subseq(seq: list[int], subseq: list[int]) -> list[int]:
    """Return all start indices where subseq occurs in seq."""
    positions = []
    n, m = len(seq), len(subseq)
    for i in range(n - m + 1):
        if seq[i:i + m] == subseq:
            positions.append(i)
    return positions


def pretokenize(raw_dataset, tokenizer, max_seq_len: int, label: str = "") -> Dataset:
    """Tokenize + build labels with response-only masking in the main process.

    The Gemma3 tokenizer/processor cannot be pickled, so we tokenize here
    (single-threaded) rather than letting TRL do it with multiprocessing.
    """
    resp_ids = tokenizer.encode("<start_of_turn>model\n", add_special_tokens=False)
    inst_ids = tokenizer.encode("<start_of_turn>user\n", add_special_tokens=False)

    all_input_ids = []
    all_labels = []

    for i, example in enumerate(raw_dataset):
        if i % 2000 == 0:
            print(f"  {label} {i}/{len(raw_dataset)}")

        text = format_to_gemma(example["messages"])
        enc = tokenizer(text, truncation=True, max_length=max_seq_len)
        ids = enc["input_ids"]

        # Build labels: copy of input_ids with -100 for non-response tokens
        labels = [-100] * len(ids)

        resp_positions = _find_subseq(ids, resp_ids)
        inst_positions = _find_subseq(ids, inst_ids)

        boundaries = []
        for p in resp_positions:
            boundaries.append((p + len(resp_ids), "on"))
        for p in inst_positions:
            boundaries.append((p, "off"))
        boundaries.sort(key=lambda x: x[0])

        masking_on = False
        prev_pos = 0
        for pos, btype in boundaries:
            if masking_on:
                for j in range(prev_pos, min(pos, len(ids))):
                    labels[j] = ids[j]
            masking_on = (btype == "on")
            prev_pos = pos
        if masking_on:
            for j in range(prev_pos, len(ids)):
                labels[j] = ids[j]

        all_input_ids.append(ids)
        all_labels.append(labels)

    print(f"  {label} Tokenized {len(all_input_ids)} examples")
    return Dataset.from_dict({
        "input_ids": all_input_ids,
        "labels": all_labels,
        "attention_mask": [[1] * len(ids) for ids in all_input_ids],
    })


# ── Trainer subclass ─────────────────────────────────────────────────────────

class UnslothSFTTrainer(SFTTrainer):
    """SFTTrainer that skips entropy_from_logits.

    Unsloth makes outputs.logits a lazy callable to save memory.
    TRL's compute_loss tries to subscript it, which crashes.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        return _HFTrainer.compute_loss(
            self, model, inputs, return_outputs=return_outputs, **kwargs
        )


# ── Generation callback ─────────────────────────────────────────────────────

class GenerateCallback(TrainerCallback):
    """Log sample generations to wandb at each eval step."""

    def __init__(self, raw_val_examples: list[dict], tokenizer, max_new_tokens: int = 256):
        self.raw_val_examples = raw_val_examples
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None:
            return

        model.eval()
        table_rows = []

        for ex in self.raw_val_examples[:N_GENERATE_SAMPLES]:
            msgs = ex["messages"]
            prompt_text = make_prompt_prefix(msgs)

            user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
            user_short = user_msg[:300] + ("…" if len(user_msg) > 300 else "")

            expected = ""
            for m in msgs:
                if m["role"] == "assistant":
                    if m.get("tool_calls"):
                        for tc in m["tool_calls"]:
                            expected += (
                                f'[TOOL_REQUEST]{{"name": "{tc["function"]["name"]}", '
                                f'"arguments": {tc["function"]["arguments"]}}}'
                                f'[END_TOOL_REQUEST]'
                            )
                    elif m.get("content"):
                        expected = m["content"]
                    break

            inputs = self.tokenizer(
                prompt_text, return_tensors="pt", truncation=True,
                max_length=MAX_SEQ_LEN - self.max_new_tokens,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )
            generated = self.tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=False,
            )
            if "<end_of_turn>" in generated:
                generated = generated[:generated.index("<end_of_turn>")]

            table_rows.append([user_short, expected, generated.strip()])

        if table_rows:
            table = wandb.Table(
                columns=["user_input", "expected", "generated"],
                data=table_rows,
            )
            wandb.log({"val_generations": table, "global_step": state.global_step})

        model.train()


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data",   default="data/train.jsonl")
    p.add_argument("--output", default="outputs/medgemma-calc")
    p.add_argument("--epochs", type=int,   default=2)
    p.add_argument("--lr",     type=float, default=2e-4)
    p.add_argument("--batch",  type=int,   default=2,
                   help="Per-device train batch size")
    p.add_argument("--grad-accum", type=int, default=8,
                   help="Gradient accumulation (effective batch = batch × grad_accum)")
    p.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    p.add_argument("--resume", default=None,
                   help="Path to checkpoint dir to resume from")
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── 1. Load model ─────────────────────────────────────────────────────────
    print(f"Loading {MODEL_NAME} in 4-bit …")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=args.max_seq_len,
        dtype=None,
        load_in_4bit=True,
        attn_implementation="flash_attention_2",
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=True,
        loftq_config=None,
    )

    # Restore original transformers attention forward (Unsloth's SDPA patch
    # has a mask-shape bug with padded batches on Gemma3).
    _g3_module.Gemma3Attention.forward = _ORIGINAL_GEMMA3_ATTN_FORWARD

    # Freeze vision tower + multi-modal projector (text-only fine-tuning)
    for name, param in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            param.requires_grad = False

    # Unsloth returns a Processor for Gemma3; extract the underlying tokenizer
    from transformers import ProcessorMixin
    if isinstance(tokenizer, ProcessorMixin):
        text_tokenizer = tokenizer.tokenizer
    else:
        text_tokenizer = tokenizer

    # ── 2. Pre-tokenize dataset ───────────────────────────────────────────────
    print(f"Loading dataset from {args.data} …")
    raw_dataset = load_dataset("json", data_files={"train": args.data}, split="train")

    if N_VAL > 0:
        print(f"Splitting {N_VAL} examples for validation …")
        split = raw_dataset.train_test_split(test_size=N_VAL, seed=42)
        raw_train = split["train"]
        raw_val = split["test"]
    else:
        print("No validation split (N_VAL=0), using all data for training …")
        raw_train = raw_dataset
        raw_val = None

    raw_val_examples = (
        [raw_val[i] for i in range(min(N_GENERATE_SAMPLES, len(raw_val)))]
        if raw_val is not None else []
    )

    print("Tokenizing + masking in main process …")
    train_dataset = pretokenize(raw_train, text_tokenizer, args.max_seq_len, label="train")
    val_dataset = (
        pretokenize(raw_val, text_tokenizer, args.max_seq_len, label="val")
        if raw_val is not None else None
    )

    # ── 3. Trainer ────────────────────────────────────────────────────────────
    use_bf16 = torch.cuda.is_bf16_supported()

    sft_config = SFTConfig(
        max_length=args.max_seq_len,
        packing=False,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=not use_bf16,
        bf16=use_bf16,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=1,
        eval_strategy="no",
        output_dir=args.output,
        report_to="wandb",
        seed=42,
        push_to_hub=False,
        run_name="medgemma-calc-qlora",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # VLM: skip TRL's internal dataset preparation (we pre-tokenized)
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
    )

    callbacks = [GenerateCallback(raw_val_examples, text_tokenizer)] if raw_val_examples else []

    trainer = UnslothSFTTrainer(
        model=model,
        processing_class=text_tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=sft_config,
        callbacks=callbacks,
    )

    # ── 4. Train ──────────────────────────────────────────────────────────────
    wandb.init(
        project="calc-caller",
        name="medgemma-calc-qlora",
        config={
            "model": MODEL_NAME,
            "lora_r": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "use_rslora": True,
            "attn_implementation": "flash_attention_2",
            "batch_size": args.batch,
            "grad_accum": args.grad_accum,
            "epochs": args.epochs,
            "lr": args.lr,
            "max_seq_len": args.max_seq_len,
            "packing": False,
            "train_examples": len(train_dataset),
            "val_examples": len(val_dataset) if val_dataset is not None else 0,
        },
    )

    effective_batch = args.batch * args.grad_accum
    print(
        f"\nStarting QLoRA training\n"
        f"  model         : {MODEL_NAME}\n"
        f"  LoRA r/alpha  : {LORA_RANK}/{LORA_ALPHA}\n"
        f"  batch (eff.)  : {args.batch} × {args.grad_accum} = {effective_batch}\n"
        f"  epochs        : {args.epochs}\n"
        f"  packing       : False\n"
        f"  precision     : {'bf16' if use_bf16 else 'fp16'}\n"
        f"  train/val     : {len(train_dataset)}/{len(val_dataset) if val_dataset is not None else 0}\n"
        f"  output        : {args.output}\n"
    )

    trainer.train(resume_from_checkpoint=args.resume)

    # ── 5. Save ───────────────────────────────────────────────────────────────
    print(f"\nSaving adapter to {args.output} …")
    model.save_pretrained(args.output)
    text_tokenizer.save_pretrained(args.output)

    merged_dir = args.output + "-merged"
    print(f"Merging adapter into full model → {merged_dir} …")
    model.save_pretrained_merged(merged_dir, text_tokenizer, save_method="merged_16bit")

    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
