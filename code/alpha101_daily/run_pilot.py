"""Run the daily OHLCV-only Alpha101 pilot."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

if __package__:
    from .alphas import ALPHA_COLUMNS, build_alpha101
    from .engine import evaluate_alpha, prepare_execution_data
else:
    from alphas import ALPHA_COLUMNS, build_alpha101
    from engine import evaluate_alpha, prepare_execution_data


def main():
    project_dir = Path(__file__).resolve().parents[2]
    processed_dir = project_dir / "data" / "processed"
    output_dir = project_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_parquet(
        processed_dir / "factors_with_composite.parquet"
    )
    _, alpha_panel = build_alpha101(raw)
    alpha_panel.to_parquet(
        processed_dir / "alpha101_daily_raw.parquet",
        index=False,
    )

    data = raw.merge(
        alpha_panel,
        on=["date", "code"],
        how="left",
        validate="one_to_one",
    )
    data = prepare_execution_data(data, project_dir)

    summaries = []
    ic_series_parts = []
    curve_parts = []

    for alpha_column in ALPHA_COLUMNS:
        summary, detail = evaluate_alpha(
            data,
            alpha_column,
            slippage_bp=10.0,
        )
        if summary is None:
            continue
        summaries.append(summary)
        ic_part, curve_part = detail
        ic_series_parts.append(ic_part)
        curve_parts.append(curve_part)
        print(
            f"{alpha_column}: "
            f"IC={summary['IC_mean']:.4f}, "
            f"净年化={summary['net_annual_return']:.2%}"
        )

    summary = (
        pd.DataFrame(summaries)
        .sort_values(
            "train_net_sharpe",
            ascending=False,
            na_position="last",
        )
        .reset_index(drop=True)
    )
    ic_series = pd.concat(ic_series_parts, ignore_index=True)
    curves = pd.concat(curve_parts, ignore_index=True)

    summary.to_csv(
        output_dir / "alpha101_daily_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    ic_series.to_csv(
        output_dir / "alpha101_daily_ic_series.csv",
        index=False,
        encoding="utf-8-sig",
    )
    curves.to_parquet(
        processed_dir / "alpha101_daily_nav.parquet",
        index=False,
    )

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(13, 7))
    for factor in summary.head(10)["factor"]:
        part = curves[curves["factor"] == factor]
        ax.plot(part["date"], part["nav"], linewidth=1, label=factor)
    ax.set_title("Alpha101 日频因子：训练期净夏普前 10 名（探索性）")
    ax.set_xlabel("信号日期")
    ax.set_ylabel("净累计净值")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(
        output_dir / "alpha101_daily_top10.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close()

    print("\n汇总")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
