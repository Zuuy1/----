from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


project_dir = Path(__file__).resolve().parents[1]
processed_dir = project_dir / "data" / "processed"
output_dir = project_dir / "output"
output_dir.mkdir(parents=True, exist_ok=True)

min_names = 50
min_listing_bdays = 250
commission_bp = 2.5
transfer_bp = 0.1
slippage_levels_bp = [0, 5, 10, 20]
stamp_cut_date = pd.Timestamp("2023-08-28")

df = pd.read_parquet(
    processed_dir / "factors_with_composite.parquet"
)
stock_basic = pd.read_parquet(
    project_dir / "data" / "raw" / "stock_basic.parquet",
    columns=["code", "ipoDate"],
)
trading_calendar = pd.read_parquet(
    project_dir / "data" / "raw" / "trading_calendar.parquet",
)
trading_calendar = (
    trading_calendar["date"]
    .drop_duplicates()
    .sort_values()
    .to_numpy(dtype="datetime64[ns]")
)

df = df.merge(stock_basic, on="code", how="left", validate="many_to_one")
df["ipoDate"] = pd.to_datetime(df["ipoDate"], errors="coerce")
df = df.sort_values(["code", "date"]).reset_index(drop=True)

shift_fields = [
    "date",
    "open",
    "high",
    "low",
    "pctChg",
    "tradestatus",
    "isST",
]
grouped = df.groupby("code", group_keys=False)
for field in shift_fields:
    df[f"entry_{field}"] = grouped[field].shift(-1)
    df[f"exit_{field}"] = grouped[field].shift(-2)

listing_date_available = (
    df["ipoDate"].notna() & df["entry_date"].notna()
)
listing_trading_days = pd.Series(np.nan, index=df.index, dtype=float)
entry_positions = np.searchsorted(
    trading_calendar,
    df.loc[listing_date_available, "entry_date"].to_numpy(
        dtype="datetime64[ns]"
    ),
    side="left",
)
ipo_positions = np.searchsorted(
    trading_calendar,
    df.loc[listing_date_available, "ipoDate"].to_numpy(
        dtype="datetime64[ns]"
    ),
    side="left",
)
listing_trading_days.loc[listing_date_available] = (
    entry_positions - ipo_positions + 1
)
df["listing_trading_days"] = listing_trading_days

entry_star = df["code"].str.startswith("sh.688")
entry_chinext = df["code"].str.startswith("sz.30")
entry_after_reform = df["entry_date"] >= pd.Timestamp("2020-08-24")

entry_limit_pct = np.select(
    [
        df["entry_isST"].eq(1).fillna(False).to_numpy(dtype=bool),
        entry_star.fillna(False).to_numpy(dtype=bool),
        (entry_chinext & entry_after_reform)
        .fillna(False)
        .to_numpy(dtype=bool),
    ],
    [5.0, 20.0, 20.0],
    default=10.0,
)
exit_star = df["code"].str.startswith("sh.688")
exit_chinext = df["code"].str.startswith("sz.30")
exit_after_reform = df["exit_date"] >= pd.Timestamp("2020-08-24")
exit_limit_pct = np.select(
    [
        df["exit_isST"].eq(1).fillna(False).to_numpy(dtype=bool),
        exit_star.fillna(False).to_numpy(dtype=bool),
        (exit_chinext & exit_after_reform)
        .fillna(False)
        .to_numpy(dtype=bool),
    ],
    [5.0, 20.0, 20.0],
    default=10.0,
)

entry_open_high_locked = (
    np.isclose(df["entry_open"], df["entry_high"], atol=1e-8)
    & (
        df["entry_pctChg"]
        >= pd.Series(entry_limit_pct, index=df.index) - 0.1
    )
)
entry_open_low_locked = (
    np.isclose(df["entry_open"], df["entry_low"], atol=1e-8)
    & (
        df["entry_pctChg"]
        <= -pd.Series(entry_limit_pct, index=df.index) + 0.1
    )
)
exit_open_high_locked = (
    np.isclose(df["exit_open"], df["exit_high"], atol=1e-8)
    & (
        df["exit_pctChg"]
        >= pd.Series(exit_limit_pct, index=df.index) - 0.1
    )
)
exit_open_low_locked = (
    np.isclose(df["exit_open"], df["exit_low"], atol=1e-8)
    & (
        df["exit_pctChg"]
        <= -pd.Series(exit_limit_pct, index=df.index) + 0.1
    )
)

factor_valid = df["COMPOSITE_Z"].notna()
entry_exit_available = (
    df["tradestatus"].eq(1)
    & df["entry_date"].notna()
    & df["exit_date"].notna()
    & (df["entry_open"] > 0)
    & (df["exit_open"] > 0)
    & df["entry_tradestatus"].eq(1)
    & df["exit_tradestatus"].eq(1)
)
not_st = (
    df["isST"].eq(0)
    & df["entry_isST"].eq(0)
    & df["exit_isST"].eq(0)
)
listing_age_ok = df["listing_trading_days"] >= min_listing_bdays
not_limit_locked = ~(
    entry_open_high_locked.fillna(False)
    | entry_open_low_locked.fillna(False)
    | exit_open_high_locked.fillna(False)
    | exit_open_low_locked.fillna(False)
)

eligible_mask = (
    factor_valid
    & entry_exit_available
    & not_st
    & listing_age_ok
    & not_limit_locked
).fillna(False)
eligible = df.loc[eligible_mask].copy()

if eligible[["isST", "entry_isST", "exit_isST"]].fillna(1).ne(0).any().any():
    raise ValueError("可交易样本仍包含 ST 记录")
if not (eligible["entry_date"] > eligible["date"]).all():
    raise ValueError("存在不晚于信号日的执行日期")
if not (eligible["exit_date"] > eligible["entry_date"]).all():
    raise ValueError("存在不晚于入场日的出场日期")
if eligible["listing_trading_days"].min() < min_listing_bdays:
    raise ValueError("可交易样本违反上市天数限制")

day_count = eligible.groupby("date")["code"].transform("size")
eligible = eligible[day_count >= min_names].copy()
eligible["layer"] = eligible.groupby(
    "date",
    group_keys=False,
)["COMPOSITE_Z"].transform(
    lambda s: pd.qcut(
        s.rank(method="first"),
        q=5,
        labels=["G1", "G2", "G3", "G4", "G5"],
    )
)
eligible["exec_ret"] = (
    eligible["exit_open"] / eligible["entry_open"] - 1
)

long_positions = eligible[eligible["layer"] == "G5"].copy()
layer_count = long_positions.groupby("date")["code"].transform("size")
long_positions["weight"] = 1 / layer_count

weights = (
    long_positions.pivot_table(
        index="date",
        columns="code",
        values="weight",
        aggfunc="sum",
    )
    .fillna(0.0)
    .sort_index()
)
returns = (
    long_positions.pivot_table(
        index="date",
        columns="code",
        values="exec_ret",
        aggfunc="last",
    )
    .reindex(index=weights.index, columns=weights.columns)
)

held_missing = int((weights.ne(0) & returns.isna()).sum().sum())
if held_missing:
    raise ValueError(f"持仓有 {held_missing} 个缺失的执行收益")

returns_for_drift = returns.fillna(0.0)
gross_ret = (weights * returns_for_drift).sum(axis=1)
drifted_weights = (
    weights
    .mul(1 + returns_for_drift)
    .div(1 + gross_ret, axis=0)
)

buy_values = np.empty(len(weights))
sell_values = np.empty(len(weights))
buy_values[0] = weights.iloc[0].clip(lower=0).sum()
sell_values[0] = weights.iloc[0].clip(upper=0).abs().sum()

delta = weights.iloc[1:].to_numpy() - drifted_weights.iloc[:-1].to_numpy()
buy_values[1:] = np.clip(delta, 0, None).sum(axis=1)
sell_values[1:] = np.clip(-delta, 0, None).sum(axis=1)

execution_dates = (
    eligible.groupby("date")["entry_date"].first().reindex(weights.index)
)
if execution_dates.isna().any():
    raise ValueError("存在无法确定的执行日期")

trades = pd.DataFrame({
    "signal_date": weights.index,
    "execution_date": execution_dates.values,
    "gross_ret": gross_ret.values,
    "buy_notional": buy_values,
    "sell_notional": sell_values,
})
trades["is_initial"] = np.r_[True, np.zeros(len(trades) - 1, dtype=bool)]
trades["turnover"] = (
    trades["buy_notional"] + trades["sell_notional"]
) / 2
trades["stamp_bp"] = np.where(
    trades["execution_date"] < stamp_cut_date,
    10.0,
    5.0,
)

scenario_rows = []
curve_frames = []

for slippage_bp in slippage_levels_bp:
    traded = trades["buy_notional"] + trades["sell_notional"]
    commission_cost = traded * commission_bp / 1e4
    transfer_cost = traded * transfer_bp / 1e4
    stamp_cost = trades["sell_notional"] * trades["stamp_bp"] / 1e4
    slippage_cost = traded * slippage_bp / 1e4
    total_cost = (
        commission_cost
        + transfer_cost
        + stamp_cost
        + slippage_cost
    )
    net_ret = trades["gross_ret"] - total_cost
    nav = (1 + net_ret).cumprod()
    drawdown = nav / nav.cummax() - 1

    scenario_rows.append({
        "strategy": "long_only_G5",
        "slippage_bp": slippage_bp,
        "commission_bp": commission_bp,
        "transfer_bp": transfer_bp,
        "stamp_rule": "10bp before 2023-08-28, 5bp after",
        "days": len(net_ret),
        "cumulative_return": nav.iloc[-1] - 1,
        "annual_return": nav.iloc[-1] ** (252 / len(net_ret)) - 1,
        "annual_vol": net_ret.std(ddof=1) * np.sqrt(252),
        "sharpe": (
            net_ret.mean()
            / net_ret.std(ddof=1)
            * np.sqrt(252)
        ),
        "max_drawdown": drawdown.min(),
        "commission_cost_total": commission_cost.sum(),
        "transfer_cost_total": transfer_cost.sum(),
        "stamp_cost_total": stamp_cost.sum(),
        "slippage_cost_total": slippage_cost.sum(),
        "total_cost": total_cost.sum(),
        "avg_daily_turnover": trades["turnover"].mean(),
    })
    curve_frames.append(pd.DataFrame({
        "signal_date": trades["signal_date"],
        "execution_date": trades["execution_date"],
        "slippage_bp": slippage_bp,
        "gross_ret": trades["gross_ret"],
        "commission_cost": commission_cost,
        "transfer_cost": transfer_cost,
        "stamp_cost": stamp_cost,
        "slippage_cost": slippage_cost,
        "total_cost": total_cost,
        "net_ret": net_ret,
        "nav": nav,
    }))

summary = pd.DataFrame(scenario_rows)
curves = pd.concat(curve_frames, ignore_index=True)

if summary.isna().any().any() or curves.isna().any().any():
    raise ValueError("可执行回测结果存在缺失值")

quality = pd.DataFrame([{
    "factor_valid_rows": int(factor_valid.sum()),
    "entry_exit_available_rows": int((factor_valid & entry_exit_available).sum()),
    "not_st_rows": int((factor_valid & entry_exit_available & not_st).sum()),
    "listing_age_ok_rows": int((factor_valid & entry_exit_available & not_st & listing_age_ok).sum()),
    "not_limit_locked_rows": int((factor_valid & entry_exit_available & not_st & listing_age_ok & not_limit_locked).sum()),
    "eligible_rows": len(eligible),
    "trade_dates": len(weights),
    "weight_codes": weights.shape[1],
    "avg_holdings": float(weights.ne(0).sum(axis=1).mean()),
    "held_missing_returns": held_missing,
    "gross_exposure_mean": float(weights.abs().sum(axis=1).mean()),
    "net_exposure_max_abs": float(weights.sum(axis=1).abs().max()),
    "avg_turnover": float(trades["turnover"].mean()),
    "initial_turnover": float(trades["turnover"].iloc[0]),
    "first_signal_date": weights.index.min().date().isoformat(),
    "last_signal_date": weights.index.max().date().isoformat(),
}])

quality.to_csv(
    output_dir / "long_only_quality.csv",
    index=False,
    encoding="utf-8-sig",
)
summary.to_csv(
    output_dir / "long_only_cost_summary.csv",
    index=False,
    encoding="utf-8-sig",
)
trades.to_parquet(
    processed_dir / "long_only_trades.parquet",
    index=False,
)
curves.to_parquet(
    processed_dir / "long_only_net_nav.parquet",
    index=False,
)

long_short_summary_path = output_dir / "executable_cost_summary.csv"
if long_short_summary_path.exists():
    long_short_summary = pd.read_csv(long_short_summary_path)
    long_short_summary.insert(0, "strategy", "long_short_G5_G1")
    comparison = pd.concat(
        [
            long_short_summary,
            summary,
        ],
        ignore_index=True,
    )
    comparison.to_csv(
        output_dir / "long_only_vs_long_short.csv",
        index=False,
        encoding="utf-8-sig",
    )

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(12, 6))
for slippage_bp, group in curves.groupby("slippage_bp", sort=True):
    ax.plot(
        group["execution_date"],
        group["nav"],
        linewidth=1,
        label=f"滑点 {slippage_bp} bp",
    )
ax.set_title("可执行口径：COMPOSITE G5 单边做多净净值")
ax.set_xlabel("执行日期")
ax.set_ylabel("累计净值")
ax.grid(alpha=0.25)
ax.legend()
plt.tight_layout()
plt.savefig(
    output_dir / "long_only_net_nav.png",
    dpi=160,
    bbox_inches="tight",
)
plt.close()

long_short_curve_path = processed_dir / "executable_net_nav.parquet"
if long_short_curve_path.exists():
    long_short_curves = pd.read_parquet(long_short_curve_path)
    fig, ax = plt.subplots(figsize=(12, 6))

    for slippage_bp, group in curves[
        curves["slippage_bp"].isin([0, 10])
    ].groupby("slippage_bp", sort=True):
        ax.plot(
            group["execution_date"],
            group["nav"],
            linewidth=1.2,
            label=f"单边做多，滑点 {slippage_bp} bp",
        )

    for slippage_bp, group in long_short_curves[
        long_short_curves["slippage_bp"].isin([0, 10])
    ].groupby("slippage_bp", sort=True):
        ax.plot(
            group["execution_date"],
            group["nav"],
            linewidth=1,
            linestyle="--",
            label=f"多空，滑点 {slippage_bp} bp",
        )

    ax.set_title("单边做多与多空方案对比")
    ax.set_xlabel("执行日期")
    ax.set_ylabel("累计净值")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        output_dir / "long_only_vs_long_short.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close()

print("单边做多质量检查")
print(quality.round(6).to_string(index=False))
print("\n成本情景")
print(summary.round(4).to_string(index=False))
