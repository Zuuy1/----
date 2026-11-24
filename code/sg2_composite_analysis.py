from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 项目路径
project_dir = Path(__file__).resolve().parents[1]
processed_dir = project_dir / "data" / "processed"
output_dir = project_dir / "output"
output_dir.mkdir(parents=True, exist_ok=True)

input_path = processed_dir / "factors_standardized.parquet"

# 根据当前样本 IC 方向对齐：
# BP 正向，MOM20 反向，VOL60 反向。
# 报告中需要注明这是基于样本内 IC 的方向选择。
directions = {
    "BP": 1.0,
    "MOM20": -1.0,
    "VOL60": -1.0,
}

factors = {
    "BP": "BP_Z",
    "MOM20": "MOM20_Z",
    "VOL60": "VOL60_Z",
    "COMPOSITE": "COMPOSITE_Z",
}

portfolios = ["G1", "G2", "G3", "G4", "G5", "LS"]


def robust_daily_rank_ic(day, factor_column, min_obs=30):
    sample = day.dropna(
        subset=[
            factor_column,
            "next_ret",
            "tradestatus",
            "next_tradestatus",
        ]
    )

    sample = sample[
        (sample["tradestatus"] == 1)
        & (sample["next_tradestatus"] == 1)
    ]

    if len(sample) < min_obs:
        return np.nan

    return sample[factor_column].corr(
        sample["next_ret"],
        method="spearman",
    )


def compound_return(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float((1 + values).prod() - 1)


# 读取标准化后的因子
df = pd.read_parquet(input_path)
df = df.sort_values(["code", "date"]).reset_index(drop=True)

# 下一交易日交易状态
df["next_tradestatus"] = (
    df.groupby("code", group_keys=False)["tradestatus"].shift(-1)
)

# 计算等权合成分
# 任一因子缺失时，合成分也为缺失值
df["COMPOSITE_Z"] = (
    df["BP_Z"] * directions["BP"]
    + df["MOM20_Z"] * directions["MOM20"]
    + df["VOL60_Z"] * directions["VOL60"]
)

df.to_parquet(
    processed_dir / "factors_with_composite.parquet",
    index=False,
)


# 每日 Rank IC
ic_records = []

for date, day in df.groupby("date", sort=True):
    record = {"date": date}

    for factor_name, factor_column in factors.items():
        record[f"{factor_name}_IC"] = robust_daily_rank_ic(
            day,
            factor_column,
        )

    ic_records.append(record)

ic_series = pd.DataFrame(ic_records)

ic_series.to_csv(
    output_dir / "ic_series_with_composite.csv",
    index=False,
    encoding="utf-8-sig",
)

# IC/IR 汇总
ic_summary_rows = []

for factor_name in factors:
    ic = pd.to_numeric(
        ic_series[f"{factor_name}_IC"],
        errors="coerce",
    ).dropna()

    ic_mean = ic.mean()
    ic_std = ic.std(ddof=1)
    ir = ic_mean / ic_std if ic_std > 0 else np.nan

    ic_summary_rows.append({
        "factor": factor_name,
        "days": len(ic),
        "IC_mean": ic_mean,
        "IC_std": ic_std,
        "IR": ir,
        "positive_rate": (ic > 0).mean() if len(ic) else np.nan,
    })

ic_summary = pd.DataFrame(ic_summary_rows)

ic_summary.to_csv(
    output_dir / "ic_summary_with_composite.csv",
    index=False,
    encoding="utf-8-sig",
)


# 年度 IC
ic_long = ic_series.melt(
    id_vars="date",
    var_name="factor",
    value_name="IC",
)

ic_long["factor"] = ic_long["factor"].str.replace(
    "_IC",
    "",
    regex=False,
)
ic_long["year"] = pd.to_datetime(ic_long["date"]).dt.year

yearly_ic = (
    ic_long
    .groupby(["factor", "year"])["IC"]
    .agg(["mean", "std", "count"])
    .reset_index()
    .rename(columns={
        "mean": "IC_mean",
        "std": "IC_std",
        "count": "days",
    })
)

yearly_ic.to_csv(
    output_dir / "yearly_ic_with_composite.csv",
    index=False,
    encoding="utf-8-sig",
)


# 每日横截面因子相关性
factor_columns = ["BP_Z", "MOM20_Z", "VOL60_Z"]
correlation_records = []

for date, day in df.groupby("date", sort=True):
    sample = day[factor_columns].dropna()

    if len(sample) < 30:
        continue

    for left, right in combinations(factor_columns, 2):
        correlation_records.append({
            "date": date,
            "pair": f"{left}-{right}",
            "correlation": sample[left].corr(
                sample[right],
                method="spearman",
            ),
        })

daily_factor_correlation = pd.DataFrame(correlation_records)

factor_correlation = (
    daily_factor_correlation
    .groupby("pair")["correlation"]
    .agg(["mean", "median", "std", "count"])
    .reset_index()
    .rename(columns={
        "mean": "mean_correlation",
        "median": "median_correlation",
        "std": "correlation_std",
        "count": "days",
    })
)

factor_correlation.to_csv(
    output_dir / "factor_correlation.csv",
    index=False,
    encoding="utf-8-sig",
)


# 每个交易日分成 G1-G5
layer_records = []

for date, day in df.groupby("date", sort=True):
    for factor_name, factor_column in factors.items():
        sample = day.dropna(
            subset=[
                factor_column,
                "next_ret",
                "tradestatus",
                "next_tradestatus",
            ]
        ).copy()

        sample = sample[
            (sample["tradestatus"] == 1)
            & (sample["next_tradestatus"] == 1)
        ]

        if len(sample) < 25:
            continue

        sample["layer"] = pd.qcut(
            sample[factor_column].rank(method="first"),
            q=5,
            labels=["G1", "G2", "G3", "G4", "G5"],
        )

        grouped = sample.groupby(
            "layer",
            observed=True,
        )["next_ret"].agg(["mean", "count"])

        for layer, row in grouped.iterrows():
            layer_records.append({
                "date": date,
                "factor": factor_name,
                "layer": str(layer),
                "ret": float(row["mean"]),
                "n_stocks": int(row["count"]),
            })

layer_returns = pd.DataFrame(layer_records)

layer_returns.to_parquet(
    processed_dir / "layer_returns_with_composite.parquet",
    index=False,
)

# 长表转宽表，并计算 G5-G1
layer_returns_wide = (
    layer_returns
    .pivot(
        index=["date", "factor"],
        columns="layer",
        values="ret",
    )
    .reset_index()
)

layer_returns_wide.columns.name = None
layer_returns_wide["LS"] = (
    layer_returns_wide["G5"] - layer_returns_wide["G1"]
)

layer_returns_wide = (
    layer_returns_wide
    .sort_values(["factor", "date"])
    .reset_index(drop=True)
)

layer_returns_wide.to_parquet(
    processed_dir / "layer_returns_with_composite_wide.parquet",
    index=False,
)


# 绩效指标
performance_rows = []

for factor_name in factors:
    factor_data = (
        layer_returns_wide
        .loc[layer_returns_wide["factor"] == factor_name]
        .sort_values("date")
        .set_index("date")
    )

    for portfolio in portfolios:
        daily_ret = pd.to_numeric(
            factor_data[portfolio],
            errors="coerce",
        ).dropna()

        if daily_ret.empty:
            continue

        nav = (1 + daily_ret).cumprod()

        annual_return = (
            nav.iloc[-1] ** (252 / len(daily_ret)) - 1
        )
        annual_vol = daily_ret.std(ddof=1) * np.sqrt(252)

        daily_std = daily_ret.std(ddof=1)
        sharpe = (
            daily_ret.mean() / daily_std * np.sqrt(252)
            if daily_std > 0
            else np.nan
        )

        drawdown = nav / nav.cummax() - 1

        performance_rows.append({
            "factor": factor_name,
            "portfolio": portfolio,
            "days": len(daily_ret),
            "cumulative_return": nav.iloc[-1] - 1,
            "annual_return": annual_return,
            "annual_vol": annual_vol,
            "sharpe": sharpe,
            "max_drawdown": drawdown.min(),
            "positive_rate": (daily_ret > 0).mean(),
        })

performance = pd.DataFrame(performance_rows)

performance.to_csv(
    output_dir / "performance_table_with_composite.csv",
    index=False,
    encoding="utf-8-sig",
)


# 年度分层复合收益
yearly_layer_returns = (
    layer_returns_wide
    .assign(year=lambda x: pd.to_datetime(x["date"]).dt.year)
    .groupby(["factor", "year"])[portfolios]
    .agg(compound_return)
    .reset_index()
)

yearly_layer_returns.to_csv(
    output_dir / "yearly_layer_returns_with_composite.csv",
    index=False,
    encoding="utf-8-sig",
)


# 分层累计净值图
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(
    len(factors),
    1,
    figsize=(13, 14),
    sharex=True,
)

for ax, factor_name in zip(axes, factors):
    factor_data = (
        layer_returns_wide
        .loc[layer_returns_wide["factor"] == factor_name]
        .sort_values("date")
        .set_index("date")
    )

    for portfolio in ["G1", "G2", "G3", "G4", "G5"]:
        daily_ret = pd.to_numeric(
            factor_data[portfolio],
            errors="coerce",
        ).dropna()

        nav = (1 + daily_ret).cumprod()
        ax.plot(nav.index, nav, linewidth=1, label=portfolio)

    ls_ret = pd.to_numeric(
        factor_data["LS"],
        errors="coerce",
    ).dropna()

    ls_nav = (1 + ls_ret).cumprod()

    ax.plot(
        ls_nav.index,
        ls_nav,
        color="black",
        linewidth=1.5,
        linestyle="--",
        label="G5-G1",
    )

    ax.set_title(f"{factor_name}：分层累计净值")
    ax.set_ylabel("累计净值")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3)

axes[-1].set_xlabel("日期")

plt.tight_layout()
plt.savefig(
    output_dir / "layered_returns_with_composite.png",
    dpi=160,
    bbox_inches="tight",
)
plt.close()


# IC 时序图
fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

for ax, factor_name in zip(axes.flat, factors):
    ax.plot(
        ic_series["date"],
        ic_series[f"{factor_name}_IC"],
        linewidth=0.8,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"{factor_name} Rank IC")
    ax.grid(alpha=0.25)

plt.tight_layout()
plt.savefig(
    output_dir / "ic_series_with_composite.png",
    dpi=160,
    bbox_inches="tight",
)
plt.close()


print("\nIC 汇总")
print(ic_summary.round(4).to_string(index=False))

print("\n因子相关性")
print(factor_correlation.round(4).to_string(index=False))

print("\n绩效汇总")
print(performance.round(4).to_string(index=False))
