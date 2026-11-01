#!/usr/bin/env python3
"""Single-prompt quality/speed sweep of CUBIC and baselines for the interactive project-page figure.

Every condition is one SLURM job launched through each repo's launch.py. Speedup is later
normalized by the uncached run on the same code path; fidelity is computed against the
uncached output by autocode/summarize_demo_tradeoff.py.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

CUBIC = Path(__file__).resolve().parents[1]
JANO = CUBIC.parent / "Jano"
PYTHON = CUBIC / ".venv/bin/python"
MODELS = {
    "flux": "/home/chenyy/WORK/models/Flux-1",
    "wan": "/home/chenyy/WORK/models/Wan2.1-T2V-1.3B",
    "firered": "/home/chenyy/WORK/models/firered-image-edit",
}
PROMPTS = {
    "flux": "A photorealistic cute cat, wearing a simple blue shirt, standing against a clear sky background.",
    "wan": "A couple in formal evening wear going home get caught in a heavy downpour with umbrellas, zoom in",
    "firered": "change orange cat to a white cat, add a dog beside it, change sitting cat to standing.",
}
FIRERED_IMAGE = str(CUBIC / "firered_exp/examples/cat.png")
WAN_COMMON = ["--task", "t2v-1.3B", "--size", "832*480", "--frame_num", "81", "--sample_steps", "50", "--base_seed", "42"]


def flux_conditions():
    p = PROMPTS["flux"]
    common = ["--prompt", p]
    conds = [("ori", CUBIC, "ori", None, common + ["--seed", "42", "--num_images", "1", "--warmup_runs", "1"], {})]
    for t in ("0.25", "0.4", "0.6"):
        conds.append((f"teacache_{t}", CUBIC, "teacache", None,
                      common + ["--seeds", "42", "42", "--rel_l1_thresh", t, "--disable_quality_eval"], {}))
    for r in ("2", "3", "5"):
        conds.append((f"pab_{r}", CUBIC, "pab", None,
                      common + ["--seeds", "42", "42", "--pab_skip_range", r, "--disable_quality_eval"], {}))
    conds.append(("taylorseer", CUBIC, "taylorseer", None, common + ["--cache_intervals", "2", "3", "5"], {}))
    for r in ("3", "4", "5", "6", "8"):
        conds.append((f"cubic_r{r}", CUBIC, "cubic", "autocode/flux_anchor_granularity_matrix.py",
                       ["--prompt", p, "--seeds", "42", "--stages", "frontier", "--frontier_tolerances", "0.0032",
                        "--warmup_steps", "7", "--cooldown_steps", "2", "--warmup_runs", "1",
                        "--target_speedup", r], {"output_flag": "--output_root"}))
    jano_common = ["--prompt", p, "--model-path", MODELS["flux"], "--seed", "42", "--num-images", "1", "--warmup-runs", "1"]
    conds.append(("jano_ori", JANO, "ori", None, jano_common, {"output_flag": "--output-dir"}))
    for name, extra in (("jano", []), ("jano_fast", ["--static-interval", "8", "--medium-interval", "4"])):
        conds.append((name, JANO, "jano", None, jano_common + extra, {"output_flag": "--output-dir", "after": "jano_ori"}))
    return conds


def wan_conditions():
    common = WAN_COMMON + ["--prompt", PROMPTS["wan"]]
    conds = [("ori", CUBIC, "ori", None, common, {})]
    for t in ("0.1", "0.2", "0.3"):
        conds.append((f"teacache_{t}", CUBIC, "teacache", None, common + ["--teacache_thresh", t], {}))
    for s, c in (("2", "4"), ("3", "6"), ("4", "8")):
        conds.append((f"pab_s{s}c{c}", CUBIC, "pab", None, common + ["--pab_self_range", s, "--pab_cross_range", c], {}))
    for r in ("0.3", "0.2", "0.1"):
        conds.append((f"toca_{r}", CUBIC, "tocache", None, common, {"env": {"TOCA_FRESH_RATIO": r}}))
    for r in ("2", "4", "6", "8"):
        conds.append((f"cubic_r{r}", CUBIC, "cubic", None,
                      common + ["--target_speedup", r, "--warmup_steps", "11", "--cooldown_steps", "1"], {}))
    jano_args = ["--jano_warmup", "6", "--jano_cooldown", "4", "--static_thresh", "0.2", "--medium_thresh", "0.6"]
    conds.append(("jano_ori", JANO, "ori", None, common, {}))
    conds.append(("jano", JANO, "jano", None, common + jano_args + ["--static_interval", "6", "--medium_interval", "3"], {}))
    conds.append(("jano_fast", JANO, "jano", None, common + jano_args + ["--static_interval", "8", "--medium_interval", "4"], {}))
    return conds


def firered_conditions():
    common = ["--prompt", PROMPTS["firered"], "--input_image", FIRERED_IMAGE, "--seed", "49"]
    q = ["--disable_quality_eval"]
    conds = [("ori", CUBIC, "ori", None, common, {})]
    for t in ("0.2", "0.4", "0.6"):
        conds.append((f"teacache_{t}", CUBIC, "teacache", None, common + q + ["--rel_l1_thresh", t], {}))
    for r in ("4", "8", "16"):
        conds.append((f"pab_{r}", CUBIC, "pab", None, common + q + ["--pab_skip_range", r], {}))
    for r in ("0.3", "0.2", "0.1"):
        conds.append((f"toca_{r}", CUBIC, "tocache", None, common + q + ["--tocache_fresh_ratio", r], {}))
    for name, extra in (("jano", []), ("jano_fast", ["--jano_static_interval", "8", "--jano_medium_interval", "4"])):
        conds.append((name, CUBIC, "jano", None, common + q + extra, {}))
    for s, c in (("2", "6"), ("3", "8"), ("4", "12"), ("6", "12")):
        conds.append((f"cubic_s{s}c{c}", CUBIC, "cubic", None,
                      common + ["--cubic_target_speedup", s, "--cubic_cond_target_speedup", c, "--disable_visualization"], {}))
    return conds


BUILDERS = {"flux": flux_conditions, "wan": wan_conditions, "firered": firered_conditions}


def command(model, name, repo, method, script, args, opts, out):
    gpu = ["--gpus", "1"] if repo == CUBIC else ["--gpus-per-node", "1"]
    cmd = [str(PYTHON), "launch.py", "--model", model, "--method", method, "--launcher", "srun",
           "--partition", "debug", *gpu, "--job-name", f"demo-{model}-{name}"]
    if script:
        cmd += ["--script", script]
    if repo == CUBIC:
        cmd += ["--model-path", MODELS[model]]
    return cmd + ["--", *args, opts.get("output_flag", "--output_dir"), str(out)]


def run(model, cond, root, logs):
    name, repo, method, script, args, opts = cond
    out = root / name
    done = out / "DONE"
    if done.exists():
        return name, "skipped"
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({"MODEL_PATH": MODELS[model], "PYTHON_BIN": str(PYTHON), **opts.get("env", {})})
    with (logs / f"{name}.log").open("w") as handle:
        rc = subprocess.run(command(model, name, repo, method, script, args, opts, out), cwd=repo, env=env,
                            stdout=handle, stderr=subprocess.STDOUT).returncode
    if rc == 0:
        done.write_text("ok\n")
    return name, "ok" if rc == 0 else f"failed({rc})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["flux", "wan", "firered"])
    parser.add_argument("--only", nargs="*", default=None, help="Run only these condition names")
    parser.add_argument("--output_root", default=str(CUBIC / "results/demo_tradeoff"))
    parser.add_argument("--max_parallel", type=int, default=12)
    args = parser.parse_args()

    jobs = []
    for model in args.models:
        root = Path(args.output_root) / model
        logs = root / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        conds = BUILDERS[model]()
        (root / "conditions.json").write_text(json.dumps(
            [{"name": c[0], "repo": str(c[1]), "method": c[2], "script": c[3], "args": c[4],
              "env": c[5].get("env", {})} for c in conds], indent=2))
        jobs += [(model, c, root, logs) for c in conds if args.only is None or c[0] in args.only]

    with ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
        futures = {pool.submit(run, *job): job for job in jobs}
        for fut in as_completed(futures):
            name, status = fut.result()
            print(f"{futures[fut][0]} {name}: {status}", flush=True)


if __name__ == "__main__":
    main()
