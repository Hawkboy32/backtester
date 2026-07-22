"""Renders scan results into a human-readable bot_memory.txt, backed by
machine-readable JSON/CSV for any downstream automated use.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from backtester.ranking import aggregate_by_strategy, rank_combos
from backtester.scanner import ScanResultRow


def generate_report(
    rows: list[ScanResultRow],
    meta: dict,
    top_n: int = 20,
) -> str:
    ranked = rank_combos(rows)
    aggregates = aggregate_by_strategy(rows)
    errors = [r for r in rows if r.error is not None]

    lines: list[str] = []
    lines.append("Backtester Scan Report")
    lines.append("=" * 60)
    lines.append(f"Run at (UTC): {meta.get('run_at', datetime.now(timezone.utc).isoformat())}")
    lines.append(f"Universe: {meta.get('universe', '?')} ({meta.get('num_tickers', '?')} tickers)")
    lines.append(f"Date range: {meta.get('from_date', '?')} to {meta.get('to_date', '?')}")
    lines.append(f"Granularity: {meta.get('multiplier', '?')} {meta.get('timespan', '?')}")
    lines.append(f"Strategies tested: {', '.join(meta.get('strategy_names', []))}")
    if meta.get("vol_target_enabled"):
        lines.append(
            f"GARCH volatility filter + sizing: ON (target {meta.get('target_vol_ann', '?')}% annualized vol) "
            "— entries blocked in \"storm\" regime, position size scaled by forecast vol otherwise."
        )
    lines.append("")
    lines.append(
        "CAVEAT: the ticker universe is today's index constituent list applied "
        "retroactively over the historical window above. Companies that were "
        "removed/delisted from the index during that window aren't included, "
        "which biases these results toward survivors and somewhat overstates "
        "real historical performance (survivorship bias)."
    )
    lines.append("")

    lines.append("Per-strategy summary (ranked by mean Sharpe across tickers)")
    lines.append("-" * 60)
    for i, agg in enumerate(aggregates, start=1):
        lines.append(f"{i}. {agg.strategy_name}")
        lines.append(f"   Tested on {agg.num_tickers_tested} tickers ({agg.num_errors} errored/skipped)")
        lines.append(f"   Mean Sharpe: {agg.mean_sharpe:.2f}   Median Sharpe: {agg.median_sharpe:.2f}")
        lines.append(f"   Mean return: {agg.mean_return:.2%}   % tickers profitable: {agg.pct_profitable:.0%}")
        lines.append(f"   Mean max drawdown: {agg.mean_max_drawdown:.2%}   Total trades: {agg.total_trades}")
        lines.append("")

    lines.append(f"Top {top_n} individual ticker x strategy combos (by composite score)")
    lines.append("-" * 60)
    for i, row in enumerate(ranked.head(top_n).itertuples(), start=1):
        # int() guard: if ANY result row errored, pandas stores the whole
        # num_trades column as float64 (int columns can't hold NaN) — and
        # ranked rows are all non-error, so the cast is always safe here.
        lines.append(
            f"{i:2d}. {row.ticker:6s} / {row.strategy_name:20s} "
            f"score={row.score:.3f} sharpe={row.sharpe_ratio:7.2f} "
            f"return={row.total_return:7.2%} maxdd={row.max_drawdown:7.2%} "
            f"trades={int(row.num_trades):3d} winrate={row.win_rate:.0%}"
        )
    lines.append("")

    if errors:
        lines.append(f"Errors / skipped ({len(errors)})")
        lines.append("-" * 60)
        for r in errors[:50]:
            lines.append(f"- {r.ticker} / {r.strategy_name}: {r.error}")
        if len(errors) > 50:
            lines.append(f"... and {len(errors) - 50} more")
        lines.append("")

    if aggregates:
        best = aggregates[0]
        worst = aggregates[-1]
        lines.append("What this means")
        lines.append("-" * 60)
        lines.append(
            f"Best performing strategy overall: {best.strategy_name} "
            f"(mean Sharpe {best.mean_sharpe:.2f} across {best.num_tickers_tested} tickers, "
            f"{best.pct_profitable:.0%} profitable)"
        )
        lines.append(
            f"Weakest performing strategy: {worst.strategy_name} "
            f"(mean Sharpe {worst.mean_sharpe:.2f}, {worst.pct_profitable:.0%} profitable)"
        )
        lines.append(
            "These are backtest results over a limited historical window — not a "
            "guarantee of future performance. Treat this as a starting point for "
            "further validation, not a final answer."
        )

    return "\n".join(lines)


def save_report(rows: list[ScanResultRow], meta: dict, out_dir: Path | str, top_n: int = 20) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_text = generate_report(rows, meta, top_n=top_n)
    txt_path = out_dir / "bot_memory.txt"
    txt_path.write_text(report_text, encoding="utf-8")

    raw_path = out_dir / "scan_results.json"
    raw_path.write_text(json.dumps([asdict(r) for r in rows], indent=2), encoding="utf-8")

    ranked = rank_combos(rows)
    ranked_path = out_dir / "scan_results_ranked.csv"
    ranked.to_csv(ranked_path, index=False)

    return {"report": txt_path, "raw": raw_path, "ranked": ranked_path}
