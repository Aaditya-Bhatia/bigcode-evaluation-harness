#!/usr/bin/env python3
from __future__ import annotations

import gzip
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import yaml
from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "vllm_humanevalfix_runs"
DETACHED = RESULTS / "_detached"
HUMANEVALFIX_MAX_TOKENS = 1024


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def prompt_for(sample: dict) -> str:
    return sample["declaration"] + sample["buggy_solution"] + f"\nFix bugs in {sample['entry_point']}."


def api_model(config: dict) -> str:
    return str(config.get("served_model_name") or config.get("model_name") or "model")


def text_completion(api_base: str, model: str, prompt: str, config: dict) -> str:
    # POST /v1/completions (raw text continuation). Do NOT use /v1/chat/completions:
    # vLLM applies the base model's chat template before tokenization, which wraps
    # the prompt in ChatML / instruct turn markers. The HumanEvalFix dirty LoRAs are
    # trained on plain buggy_code -> fixed_code text pairs and never learn to emit
    # <|im_end|>; they degenerate to prompt-echo + max_tokens of EOS-region noise
    # (CJK glyphs on Qwen, BPE byte markers on deepseek, scaffolding on codellama).
    # The bigcode upstream harness has always used /v1/completions for this task.
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": float(config.get("temperature", 0.2)),
        "top_p": float(config.get("top_p", 0.95)),
        "max_tokens": HUMANEVALFIX_MAX_TOKENS,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        api_base.rstrip("/") + "/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'EMPTY')}",
        },
        method="POST",
    )
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["text"]
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError):
            if attempt == 3:
                raise
            time.sleep(2 * attempt)
    raise RuntimeError("unreachable")


def write_summary(path: Path, **updates) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
    data.update(updates)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_partials(path: Path) -> dict[int, dict]:
    """Read existing partials.jsonl. Returns {sample_index: row}.

    Tolerates a half-written final line (will be overwritten on next append)
    but otherwise enforces that every line is valid JSON with an int "index".
    """
    rows: dict[int, dict] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # Truncated trailing line from a previous crash; ignore.
                continue
            idx = row.get("__index")
            if isinstance(idx, int):
                rows[idx] = row
    return rows


class PartialsWriter:
    """Append-only writer for the per-sample checkpoint. Each call to
    record() appends one JSON line under a single shared lock so concurrent
    workers do not interleave bytes."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = Lock()

    def record(self, index: int, row: dict) -> None:
        out = dict(row)
        out["__index"] = int(index)
        line = json.dumps(out) + "\n"
        with self._lock:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                try:
                    os.fsync(handle.fileno())
                except OSError:
                    pass


def main() -> int:
    config_path = Path(sys.argv[1]).resolve()
    config = load_config(config_path)
    model_name = str(config.get("model_name") or "model")
    model = api_model(config)
    api_base = str(config.get("api_base") or f"http://127.0.0.1:{int(config.get('port', 8000))}/v1")

    hef = dict(config.get("humanevalfix") or {})
    resume_run_name = hef.get("resume_from_run_name")
    previous_attempts = int(hef.get("previous_attempts") or 0)

    if resume_run_name:
        run_name = str(resume_run_name)
        run_dir = DETACHED / run_name
        if not run_dir.exists():
            # Orchestrator told us to resume into a dir that's gone; fall
            # back to a fresh run so we still make progress.
            resume_run_name = None
    if not resume_run_name:
        run_name = (
            f"humanevalfixtests-python__{model_name}__instruct__"
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
            f"{time.time_ns() % 1_000_000_000:09d}"
        )
        run_dir = DETACHED / run_name

    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    generation_path = run_dir / f"{run_name}.generations.jsonl.gz"
    error_path = run_dir / "errors.jsonl"
    partials_path = run_dir / "partials.jsonl"

    summary_update = dict(
        benchmark="HumanEvalFix",
        run_name=run_name,
        run_dir=str(run_dir),
        config_path=str(config_path),
        model_name=model_name,
        api_model=model,
        api_base=api_base,
        task="humanevalfixtests-python",
        prompt="instruct",
        status="running_generation",
        generation_path=str(generation_path),
        error_path=str(error_path),
        partials_path=str(partials_path),
        attempts=previous_attempts + 1,
    )
    # On a resume start, clear the prior failure fields so the orchestrator
    # does not see stale "generation_failed" state alongside running_generation.
    if resume_run_name and summary_path.exists():
        try:
            prior = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            prior = {}
        for stale in ("generation_failed_at_utc", "error"):
            if stale in prior:
                # write_summary merges, not replaces — explicitly None them
                # out so the rewritten summary drops the failure markers.
                summary_update[stale] = None
        summary_update["last_resume_started_at_utc"] = utc_now()
    else:
        summary_update["generation_started_at_utc"] = utc_now()
        summary_update["metrics_path"] = None

    write_summary(summary_path, **summary_update)

    samples = list(load_dataset("bigcode/humanevalpack", "python", split="test"))
    max_workers = max(1, min(int(config.get("max_workers", 16)), 64))

    existing_partials = load_partials(partials_path) if resume_run_name else {}
    if existing_partials:
        print(
            f">>> Resuming run {run_name}: "
            f"{len(existing_partials)}/{len(samples)} samples already on disk",
            flush=True,
        )

    outputs: list[dict | None] = [None] * len(samples)
    for idx, row in existing_partials.items():
        if 0 <= idx < len(samples):
            outputs[idx] = {k: v for k, v in row.items() if k != "__index"}

    pending_indices = [idx for idx, _ in enumerate(samples) if outputs[idx] is None]
    errors: list[dict] = []
    writer = PartialsWriter(partials_path)

    def run_one(idx: int, sample: dict) -> tuple[int, dict]:
        prompt = prompt_for(sample)
        content = text_completion(api_base, model, prompt, config)
        row = dict(sample)
        row["prompt"] = prompt
        row["raw_generation"] = [content]
        row["generation"] = [content]
        return idx, row

    if pending_indices:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(run_one, idx, samples[idx]): idx for idx in pending_indices
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    row_idx, row = fut.result()
                    outputs[row_idx] = row
                    writer.record(row_idx, row)
                except Exception as exc:  # noqa: BLE001
                    errors.append({"index": idx, "error": repr(exc)})

    if errors:
        with error_path.open("w", encoding="utf-8") as handle:
            for item in errors:
                handle.write(json.dumps(item) + "\n")
        write_summary(
            summary_path,
            status="generation_failed",
            generation_failed_at_utc=utc_now(),
            error=f"{len(errors)} generation request(s) failed",
        )
        return 1

    with gzip.open(generation_path, "wt", encoding="utf-8") as handle:
        for row in outputs:
            handle.write(json.dumps(row) + "\n")

    write_summary(
        summary_path,
        status="generation_complete",
        generation_completed_at_utc=utc_now(),
        sample_count=len(outputs),
        error=None,
    )
    print(json.dumps(json.loads(summary_path.read_text(encoding="utf-8")), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
