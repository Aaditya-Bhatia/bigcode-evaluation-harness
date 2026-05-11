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


def chat_completion(api_base: str, model: str, prompt: str, config: dict) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(config.get("temperature", 0.2)),
        "top_p": float(config.get("top_p", 0.95)),
        "max_tokens": HUMANEVALFIX_MAX_TOKENS,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        api_base.rstrip("/") + "/chat/completions",
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
            return data["choices"][0]["message"]["content"]
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


def main() -> int:
    config_path = Path(sys.argv[1]).resolve()
    config = load_config(config_path)
    model_name = str(config.get("model_name") or "model")
    model = api_model(config)
    api_base = str(config.get("api_base") or f"http://127.0.0.1:{int(config.get('port', 8000))}/v1")
    run_name = f"humanevalfixtests-python__{model_name}__instruct__{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{time.time_ns() % 1_000_000_000:09d}"
    run_dir = DETACHED / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    generation_path = run_dir / f"{run_name}.generations.jsonl.gz"
    error_path = run_dir / "errors.jsonl"
    write_summary(
        summary_path,
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
        generation_started_at_utc=utc_now(),
        generation_path=str(generation_path),
        error_path=str(error_path),
        metrics_path=None,
    )

    samples = list(load_dataset("bigcode/humanevalpack", "python", split="test"))
    max_workers = max(1, min(int(config.get("max_workers", 16)), 64))
    outputs: list[dict | None] = [None] * len(samples)
    errors: list[dict] = []

    def run_one(idx: int, sample: dict) -> tuple[int, dict]:
        prompt = prompt_for(sample)
        content = chat_completion(api_base, model, prompt, config)
        row = dict(sample)
        row["prompt"] = prompt
        row["raw_generation"] = [content]
        row["generation"] = [content]
        return idx, row

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run_one, idx, sample): idx for idx, sample in enumerate(samples)}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                row_idx, row = fut.result()
                outputs[row_idx] = row
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
