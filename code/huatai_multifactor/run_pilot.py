"""Run the Huatai-style multi-factor pilot."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

if __package__:
    from alpha101_daily.engine import (
        evaluate_alpha,
        prepare_execution_data,
    )
    from .factors import build_huatai_factors, factor_columns
else:
    from alpha101_daily.engine import (
        evaluate_alpha,
        prepare_execution_data,
    )
    from factors import build_huatai_factors, factor_columns


def main():
    project_dir = Path(__file__).resolve().parents[2]
    processed_dir = project_dir / "data" / "processed"
    output_dir = project_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_parquet(
        processed_dir / "factors_with_composite.parquet"
    )
    data = build_huatai_factors(raw, project_dir)
    columns = factor_columns()
    data[["date", "code", *columns]].to_parquet(
        processed_dir / "huatai_multifactor_raw.parquet",
        index=False,
    )
    data = prepare_execution_data(data, project_dir)

    summaries = []
    ic_parts = []
    curve_parts = []

    for factor in columns:
        summary, detail = evaluate_alpha(
            data,
            factor,
            slippage_bp=10.0,
        )
        if summary is None:
            continue
        summaries.append(summary)
        ic_part, curve_part = detail
        ic_parts.append(ic_part)
        curve_parts.append(curve_part)
        print(
            f"{factor}: IC={summary['IC_mean']:.4f}, "
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
    ic_series = pd.concat(ic_parts, ignore_index=True)
    curves = pd.concat(curve_parts, ignore_index=True)

    summary.to_csv(
        output_dir / "huatai_multifactor_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    ic_series.to_csv(
        output_dir / "huatai_multifactor_ic_series.csv",
        index=False,
        encoding="utf-8-sig",
    )
    curves.to_parquet(
        processed_dir / "huatai_multifactor_nav.parquet",
        index=False,
    )

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(13, 7))
    for factor in summary.head(10)["factor"]:
        part = curves[curves["factor"] == factor]
        ax.plot(part["date"], part["nav"], linewidth=1, label=factor)
    ax.set_title("华泰风格多因子：训练期净夏普前 10 名")
    ax.set_xlabel("信号日期")
    ax.set_ylabel("净累计净值")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(
        output_dir / "huatai_multifactor_top10.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close()

    print("\n汇总")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
