"""Manual entry point: runs the real graph once right now and prints
everything - activity, risk, and the briefing. `--notify` actually sends
it through notifications.notify_daily_briefing; without it, nothing is
sent anywhere. Use this to test; scheduler.py is what runs continuously.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from backtester import notifications  # noqa: E402

from graph import run_briefing  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["ollama", "claude"], default="ollama")
    parser.add_argument("--model")
    parser.add_argument("--notify", action="store_true", help="Actually send the notification, not just print it.")
    args = parser.parse_args()

    result = run_briefing(provider=args.provider, model=args.model)

    print("--- activity ---")
    print(json.dumps(result["activity"], indent=2))
    print("\n--- risk ---")
    print(json.dumps(result["risk"], indent=2))
    print("\n--- briefing ---")
    print(result["briefing"])

    if args.notify:
        sent = notifications.notify_daily_briefing(result["briefing"])
        print(f"\n(notify_daily_briefing returned {sent})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
