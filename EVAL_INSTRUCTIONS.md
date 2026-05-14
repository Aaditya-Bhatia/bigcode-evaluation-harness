# Eval Instructions (detached CPU eval) — HumanEvalFix

This repo contains **generation outputs** from the same-size dirty-LoRA
experiment, committed by the k8s GPU pod. The judge / eval phase runs on
a separate CPU machine using these committed artifacts — generation is
not re-run.

## Critical: experiment isolation

The same model_name slugs (e.g. `qwen2.5-coder-7b-lora-dirty`) are
**shared** with the prior different-size experiment. The two are
distinguished ONLY by `summary.json:"config_path"`:

- **Same-size** (this experiment, what we want to eval):
  `config_path` starts with `/workspace/Master-Benchmarking-Orchestrator/`
- **Different-size** (older experiment):
  `config_path` starts with `/shared_workspace_mfs/.../Master_VLLM/`

Before running judges, verify every run dir:

```bash
for f in vllm_humanevalfix_runs/_detached/*/summary.json; do
  cfg=$(python3 -c "import json; print(json.load(open('$f')).get('config_path',''))")
  [[ "$cfg" != /workspace/Master-Benchmarking-Orchestrator/* ]] && echo "WRONG-EXPERIMENT: $f -> $cfg"
done
```

Expect: empty output. If anything prints, you have a stale checkout or
mixed history — STOP and ask before evaluating.

## Which dirs are ready

Only dirs where `summary.json:"status" == "generation_complete"`. Anything
else (`running_generation`, `generation_failed`, `generation_pending`,
`eval_running`, `eval_failed`) is either partial or already mid-judge —
skip it.

## 3B-tier models

Regenerated cleanly on 2026-05-14 and now included. Eligible 3B slugs:
`qwen2.5-3b-lora-dirty`, `qwen2.5-coder-3b-instruct-lora-dirty`,
`qwen3-4b-base-lora-dirty`, `starcoder2-3b-lora-dirty`. (The earlier
contaminated commits were force-removed from this fork before the rerun.)

`llama-3.2-3b-lora-dirty` has no generation dir on the GPU pod and is
not included; it may land later.

## Active runs

At commit time, one CanItEdit dir was still being written on the GPU
pod (`qwen3-14b-base-lora-dirty-canitedit-20260513T150232Z-...`). HEF
for that model already finished and is in this commit. CanItEdit's
14B-base run will appear in a later commit on the CanItEdit repo.

## Eval entrypoint: HumanEvalFix

Generation artifact under
`vllm_humanevalfix_runs/_detached/<run-name>/`:

- `<run-name>.generations.jsonl.gz` — 164 lines (one per HumanEvalFix
  sample). This is the single eval input.
- `summary.json` — run metadata
- `partials.jsonl` — optional partial-run checkpoint (ignored by eval)

### Prerequisites on the eval machine

- Python env with the bigcode-evaluation-harness deps
  (`pip install -e .` from the repo root, after cloning with submodules)
- The harness sandboxes test execution (sandboxed Python subprocess); no
  Docker required for the Python task

### Run eval on one model

```bash
cd /workspace/human_eval_fix/bigcode-evaluation-harness

RUN_DIR=vllm_humanevalfix_runs/_detached/<run-name>
python main.py \
    --tasks humanevalfixtests-python \
    --load_generations_path "$RUN_DIR"/<run-name>.generations.jsonl.gz \
    --allow_code_execution \
    --metric_output_path "$RUN_DIR"/metrics.json \
    --n_samples 1 \
    --batch_size 1 \
    --model any-hf-id-for-the-tokenizer
```

`--load_generations_path` triggers eval-only mode (`main.py:151`). The
`--model` arg is only used to load a tokenizer; it does NOT load the
generating model — pick any HF id (e.g. `bigcode/starcoder2-7b`) that
the eval host has cached or can download.

### Run dirs ready (as of this commit)

```
humanevalfixtests-python__codellama-13b-hf-lora-dirty__instruct__20260513T040435Z-955173078
humanevalfixtests-python__codellama-7b-hf-lora-dirty__instruct__20260511T032141Z-619142024
humanevalfixtests-python__deepseek-coder-6.7b-base-lora-dirty__instruct__20260511T233649Z-498455740
humanevalfixtests-python__qwen2.5-coder-14b-instruct-lora-dirty__instruct__20260513T085929Z-830776283
humanevalfixtests-python__qwen2.5-coder-14b-lora-dirty__instruct__20260513T082349Z-017116915
humanevalfixtests-python__qwen2.5-coder-7b-instruct-lora-dirty__instruct__20260511T032610Z-762582079
humanevalfixtests-python__qwen2.5-coder-7b-lora-dirty__instruct__20260511T032145Z-783299648
humanevalfixtests-python__qwen3-8b-base-lora-dirty__instruct__20260511T211524Z-367586136
humanevalfixtests-python__starcoder2-15b-lora-dirty__instruct__20260513T105234Z-613872658
humanevalfixtests-python__starcoder2-7b-lora-dirty__instruct__20260512T162201Z-396732922
humanevalfixtests-python__qwen2.5-3b-lora-dirty__instruct__20260514T020246Z-812395263
humanevalfixtests-python__qwen2.5-coder-3b-instruct-lora-dirty__instruct__20260514T020220Z-730212790
humanevalfixtests-python__qwen3-4b-base-lora-dirty__instruct__20260514T022131Z-338695917
humanevalfixtests-python__starcoder2-3b-lora-dirty__instruct__20260514T031603Z-649934236
```
