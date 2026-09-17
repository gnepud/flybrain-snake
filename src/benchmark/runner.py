#!/usr/bin/env python3
"""Standalone CLI entry point for FlyBrain Snake Headless Benchmark & Evaluation Suite.

Usage:
    python3 -m src.benchmark.runner [args]
    python3 src/benchmark/runner.py [args]

Options:
    --episodes INT       Number of benchmark episodes (default: 50)
    --seed INT           Base seed for reproducibility (default: 1000)
    --width INT          Grid width (default: 16)
    --height INT         Grid height (default: 16)
    --max-steps INT      Maximum steps per episode (default: 2000)
    --output PATH        Path to write JSON benchmark report
    --json               Output raw JSON to stdout
    --quiet              Minimal output mode
"""

import argparse
import json
from pathlib import Path
import sys

# Ensure repository root is on sys.path when executed directly
if __name__ == "__main__" and __package__ is None:
    root_dir = str(Path(__file__).resolve().parent.parent.parent)
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

from src.benchmark.evaluator import BenchmarkEvaluator


def format_p_val(p: float) -> str:
    """Format p-value with readable precision."""
    if p < 0.0001:
        return "< 0.0001"
    return f"{p:.4f}"


def print_summary_table(report: dict) -> None:
    """Pretty-print formatted ASCII summary table for benchmark results."""
    meta = report["benchmark_metadata"]
    c_met = report["metrics"]["connectome"]
    r_met = report["metrics"]["random_baseline"]
    stats = report["statistical_tests"]
    perf = report["performance"]
    acc = report["acceptance_criteria"]

    w, h = meta["grid_size"]
    episodes = meta["episodes"]
    seed = meta["seed_base"]
    max_steps = meta["max_steps"]

    line_len = 120
    print("=" * line_len)
    print("FLYBRAIN SNAKE HEADLESS BENCHMARK & EVALUATION SUITE".center(line_len))
    print("=" * line_len)
    print(f"Episodes: {episodes} | Base Seed: {seed} | Grid: {w}x{h} | Max Steps: {max_steps}")
    print("-" * line_len)

    header = (
        f"{'Metric':<18} "
        f"{'Connectome Agent (Mean ± Std, Med)':<34} "
        f"{'Random Baseline (Mean ± Std, Med)':<32} "
        f"{'Welch t (p-val)':<18} "
        f"{'Mann-Whitney U':<18} "
        f"{'p < 0.05':<8}"
    )
    print(header)
    print("-" * line_len)

    # Survival
    c_surv_str = f"{c_met['mean_survival']:.2f} ± {c_met['std_survival']:.2f} ({c_met['median_survival']:.1f})"
    r_surv_str = f"{r_met['mean_survival']:.2f} ± {r_met['std_survival']:.2f} ({r_met['median_survival']:.1f})"
    t_s = f"t={stats['survival']['welch_t']:.2f} ({format_p_val(stats['survival']['welch_p'])})"
    u_s = f"U={stats['survival']['mann_whitney_u']:.1f} ({format_p_val(stats['survival']['mann_whitney_p'])})"
    sig_s = "PASS" if stats["survival"]["significant"] else "FAIL"

    print(f"{'Survival (steps)':<18} {c_surv_str:<34} {r_surv_str:<32} {t_s:<18} {u_s:<18} {sig_s:<8}")

    # Food Score
    c_food_str = f"{c_met['mean_food']:.2f} ± {c_met['std_food']:.2f} ({c_met['median_food']:.1f})"
    r_food_str = f"{r_met['mean_food']:.2f} ± {r_met['std_food']:.2f} ({r_met['median_food']:.1f})"
    t_f = f"t={stats['food']['welch_t']:.2f} ({format_p_val(stats['food']['welch_p'])})"
    u_f = f"U={stats['food']['mann_whitney_u']:.1f} ({format_p_val(stats['food']['mann_whitney_p'])})"
    sig_f = "PASS" if stats["food"]["significant"] else "FAIL"

    print(f"{'Food Score':<18} {c_food_str:<34} {r_food_str:<32} {t_f:<18} {u_f:<18} {sig_f:<8}")

    # Speed
    speed_val = f"{perf['steps_per_second']:.1f} steps/s"
    speed_target = "Target: >= 50.0 steps/s"
    speed_status = "PASS" if perf["speed_threshold_met"] else "FAIL"
    print(f"{'Speed (steps/s)':<18} {speed_val:<34} {speed_target:<32} {'-':<18} {'-':<18} {speed_status:<8}")

    # Memory
    mem_val = f"{perf['memory_growth_mb']:.2f} MB"
    mem_target = "Target: < 5.0 MB (over 1000 steps)"
    mem_status = "PASS" if perf["memory_threshold_met"] else "FAIL"
    print(f"{'Memory Growth':<18} {mem_val:<34} {mem_target:<32} {'-':<18} {'-':<18} {mem_status:<8}")

    print("-" * line_len)
    print("Acceptance Criteria Verification:")
    print(f"  [{'PASS' if acc['connectome_survival_superiority'] else 'FAIL'}] Connectome Survival > Random Baseline ({c_met['mean_survival']:.2f} vs {r_met['mean_survival']:.2f})")
    print(f"  [{'PASS' if acc['survival_statistical_significance'] else 'FAIL'}] Statistical Significance on Survival (p = {stats['survival']['welch_p']:.4f} < 0.05)")
    print(f"  [{'PASS' if acc['speed_threshold_satisfied'] else 'FAIL'}] Simulation Speed >= 50.0 steps/sec ({perf['steps_per_second']:.1f} steps/s)")
    print(f"  [{'PASS' if acc['memory_stability_satisfied'] else 'FAIL'}] Continuous Memory Stability < 5.0 MB ({perf['memory_growth_mb']:.2f} MB)")
    print("-" * line_len)

    if report["passed"]:
        print("OVERALL STATUS: ALL ACCEPTANCE CRITERIA PASSED (EXIT CODE 0)".center(line_len))
    else:
        print("OVERALL STATUS: ONE OR MORE CRITERIA FAILED (EXIT CODE 1)".center(line_len))
    print("=" * line_len)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="FlyBrain Snake Headless Comparative Benchmark Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--episodes", type=int, default=50, help="Number of benchmark episodes")
    parser.add_argument("--seed", type=int, default=1000, help="Base seed for reproducibility")
    parser.add_argument("--width", type=int, default=16, help="Grid width")
    parser.add_argument("--height", type=int, default=16, help="Grid height")
    parser.add_argument("--max-steps", type=int, default=2000, help="Maximum steps per episode")
    parser.add_argument("--output", type=str, default=None, help="File path to write JSON benchmark report")
    parser.add_argument("--json", action="store_true", help="Output raw JSON to stdout")
    parser.add_argument("--quiet", action="store_true", help="Minimal output mode")

    args = parser.parse_args()

    evaluator = BenchmarkEvaluator(
        width=args.width,
        height=args.height,
        max_steps=args.max_steps,
    )

    report = evaluator.run_paired_benchmark(
        episodes=args.episodes,
        seed_base=args.seed,
        width=args.width,
        height=args.height,
        max_steps=args.max_steps,
        assert_thresholds=False,
    )

    if args.output:
        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    if args.json:
        print(json.dumps(report, indent=2))
    elif args.quiet:
        status_str = "PASS" if report["passed"] else "FAIL"
        print(f"BENCHMARK {status_str}: {args.episodes} eps | Speed: {report['performance']['steps_per_second']:.1f} s/s | Mem delta: {report['performance']['memory_growth_mb']:.2f} MB")
    else:
        print_summary_table(report)

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
