#!/usr/bin/env python
"""Run the devbugs local-agent batch (no Docker) against a remote OpenAI-compatible model.

This is the project's own lightweight, hand-written bug-fix harness used for
smoke tests and offline development. It is unrelated to the third-party
``mini-swe-agent`` used for real SWE-bench runs (see ``config/mini_swe_qwen.yaml``).
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from stage2.devbugs.agent_loop import AgentConfig, run_instance
from stage2.devbugs.catalog import get_instance, list_instance_ids, load_instance_ids_from_file
from stage2.devbugs.evaluate import write_results_json


def preflight_api(cfg: AgentConfig) -> None:
    url = f"{cfg.api_base.rstrip('/')}/models"
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {cfg.api_key}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(
            f"Cannot reach model endpoint {url} ({exc.code}): {body}\n"
            "Start serve_qwen_colab.ipynb and set MODEL_API_BASE."
        ) from exc
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"Cannot reach model endpoint {url}: {exc}\n"
            "Start serve_qwen_colab.ipynb and set MODEL_API_BASE."
        ) from exc


def run_batch(
    instance_ids: list[str],
    *,
    cfg: AgentConfig,
    output_dir: Path,
    work_root: Path | None = None,
    skip_preflight: bool = False,
) -> dict:
    if not skip_preflight:
        preflight_api(cfg)

    output_dir.mkdir(parents=True, exist_ok=True)
    sandbox_root = work_root or (output_dir / "_sandboxes")
    sandbox_root.mkdir(parents=True, exist_ok=True)

    outcomes: dict[str, bool] = {}
    summary_rows: list[dict] = []

    print(f"Running {len(instance_ids)} devbugs instances -> {output_dir}", flush=True)
    print(f"Model: {cfg.model} @ {cfg.api_base}", flush=True)

    for iid in instance_ids:
        instance = get_instance(iid)
        traj_path = output_dir / f"{iid}.traj"
        print(f"\n=== {iid} ({instance.difficulty}) ===", flush=True)
        try:
            resolved, n_steps = run_instance(
                instance,
                cfg,
                work_root=sandbox_root,
                output_traj=traj_path,
            )
        except Exception as exc:
            print(f"  ERROR: {exc}", flush=True)
            resolved = False
            n_steps = 1
            query = [
                {"role": "system", "content": "error"},
                {"role": "user", "content": instance.problem_statement},
            ]
            traj_path.write_text(
                json.dumps(
                    {
                        "trajectory": [
                            {
                                "response": f"Run failed: {exc}",
                                "thought": "",
                                "action": "",
                                "observation": str(exc),
                                "query": query,
                            }
                        ],
                        "info": {
                            "instance_id": iid,
                            "exit_status": "error",
                            "error": str(exc),
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        outcomes[iid] = resolved
        status = "RESOLVED" if resolved else "FAILED"
        print(f"  {status} in {n_steps} steps -> {traj_path.name}", flush=True)
        summary_rows.append(
            {
                "instance_id": iid,
                "difficulty": instance.difficulty,
                "resolved": resolved,
                "n_steps": n_steps,
            }
        )

    results_path = output_dir / "results.json"
    write_results_json(outcomes, results_path)

    manifest = {
        "track": "devbugs_local",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model": cfg.model,
        "api_base": cfg.api_base,
        "n_instances": len(instance_ids),
        "n_resolved": sum(1 for ok in outcomes.values() if ok),
        "n_unresolved": sum(1 for ok in outcomes.values() if not ok),
        "instances": summary_rows,
        "results_json": str(results_path),
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nDone. Results: {results_path}", flush=True)
    print(
        f"Resolved {manifest['n_resolved']}/{manifest['n_instances']} "
        f"(success rate {manifest['n_resolved'] / max(len(instance_ids), 1):.0%})",
        flush=True,
    )
    print(f"Next: python -m stage2.trajectories.ingest_batch --traj-dir {output_dir}", flush=True)
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--instances",
        type=Path,
        default=Path("config/devbugs_instances.txt"),
        help="Text file with one instance id per line",
    )
    ap.add_argument(
        "--instance-id",
        action="append",
        default=[],
        help="Run a single instance (repeatable). Overrides --instances file if set.",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write .traj files (default: data/trajectories/devbugs_run_<timestamp>)",
    )
    ap.add_argument(
        "--api-base",
        default=None,
        help="OpenAI-compatible base URL (default: env MODEL_API_BASE or localhost:8000/v1)",
    )
    ap.add_argument(
        "--api-key",
        default=None,
        help="API key (default: env MODEL_API_KEY or EMPTY)",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="Model name served by vLLM (default: env MODEL_NAME or Qwen3-8B)",
    )
    ap.add_argument("--max-steps", type=int, default=25)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--skip-preflight", action="store_true")
    args = ap.parse_args()

    import os

    api_base = args.api_base or os.environ.get("MODEL_API_BASE") or "http://localhost:8000/v1"
    api_key = args.api_key or os.environ.get("MODEL_API_KEY") or "EMPTY"
    model = args.model or os.environ.get("MODEL_NAME") or "Qwen3-8B"
    # vLLM expects the served model name, not the litellm hosted_vllm/ prefix.
    if model.startswith("hosted_vllm/"):
        model = model.split("/", 1)[1]

    if args.instance_id:
        instance_ids = args.instance_id
    else:
        instance_ids = load_instance_ids_from_file(args.instances)

    unknown = [iid for iid in instance_ids if iid not in list_instance_ids()]
    if unknown:
        raise SystemExit(f"Unknown instance ids: {unknown}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or Path(f"data/trajectories/devbugs_run_{timestamp}")

    cfg = AgentConfig(
        api_base=api_base,
        api_key=api_key,
        model=model,
        max_steps=args.max_steps,
        temperature=args.temperature,
    )
    run_batch(
        instance_ids,
        cfg=cfg,
        output_dir=output_dir,
        skip_preflight=args.skip_preflight,
    )


if __name__ == "__main__":
    main()
