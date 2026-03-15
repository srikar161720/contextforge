#!/usr/bin/env python3
"""Display cumulative API spend from data/cost_log.json.

Reads cost entries written by generate_demo_payload.py and other scripts,
then prints a summary table with per-session and cumulative totals.

Usage:
    python scripts/cost_tracker.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
COST_LOG_PATH = PROJECT_ROOT / "data" / "cost_log.json"

# Pricing reference (us.amazon.nova-2-lite-v1:0)
INPUT_PRICE_PER_1M = 0.30
OUTPUT_PRICE_PER_1M = 2.50


def main() -> None:
    """Print cost summary from the cost log."""
    if not COST_LOG_PATH.exists():
        print("No cost log found. Run a generation script first.")
        sys.exit(0)

    try:
        entries = json.loads(COST_LOG_PATH.read_text())
    except (json.JSONDecodeError, ValueError):
        print("Cost log is corrupted. Delete and regenerate.")
        sys.exit(1)

    if not entries:
        print("Cost log is empty.")
        sys.exit(0)

    # Header
    print("=" * 80)
    print("ContextForge — API Cost Tracker")
    print("=" * 80)
    print(
        f"{'Timestamp':<26s} {'Script':<28s} {'Mode':<8s} "
        f"{'Calls':>6s} {'Input':>10s} {'Output':>10s} {'Cost':>10s}"
    )
    print("-" * 80)

    total_calls = 0
    total_input = 0
    total_output = 0
    total_cost = 0.0

    for entry in entries:
        ts = entry.get("timestamp", "?")[:25]
        script = entry.get("script", "?")
        mode = entry.get("mode", "?")
        calls = entry.get("api_calls", 0)
        inp = entry.get("input_tokens", 0)
        out = entry.get("output_tokens", 0)
        cost = entry.get("cost_usd", 0.0)

        total_calls += calls
        total_input += inp
        total_output += out
        total_cost += cost

        print(
            f"{ts:<26s} {script:<28s} {mode:<8s} "
            f"{calls:>6d} {inp:>10,} {out:>10,} ${cost:>8.4f}"
        )

    # Totals
    print("-" * 80)
    print(
        f"{'TOTAL':<26s} {'':<28s} {'':<8s} "
        f"{total_calls:>6d} {total_input:>10,} {total_output:>10,} ${total_cost:>8.4f}"
    )
    print("=" * 80)

    # Pricing reference
    print(f"\nPricing: ${INPUT_PRICE_PER_1M}/1M input, ${OUTPUT_PRICE_PER_1M}/1M output")
    print(f"Model: us.amazon.nova-2-lite-v1:0 (US East 1)")

    # Verify cost calculation
    computed = (total_input * INPUT_PRICE_PER_1M + total_output * OUTPUT_PRICE_PER_1M) / 1_000_000
    if abs(computed - total_cost) > 0.01:
        print(f"\nNote: Logged cost ${total_cost:.4f} vs computed ${computed:.4f} "
              f"(difference may be due to rounding)")


if __name__ == "__main__":
    main()
