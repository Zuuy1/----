from pathlib import Path

import numpy as np
import pandas as pd


project_dir = Path(__file__).resolve().parents[1]
processed_dir = project_dir / "data" / "processed"
output_dir = project_dir / "output"
processed_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

min_names = 50

df = pd.read_parquet(processed_dir / "factors_with_composite.parquet")
df = df.sort_values(["code", "date"]).reset_index(drop=True)

if "next_tradestatus" not in df.columns:
    df["next_tradestatus"] = (
        df.groupby("code", group_keys=False)["tradestatus"].shift(-1)
    )

use = df.dropna(
    subset=[
        "COMPOSITE_Z",
        "next_ret",
        "tradestatus",
        "next_tradestatus",
    ]
).copy()
use = use[
    (use["tradestatus"] == 1)
    & (use["next_tradestatus"] == 1)
].copy()

day_count = use.groupby("date")["code"].transform("size")
use = use[day_count >= min_names].copy()
use["layer"] = use.groupby("date", group_keys=False)["COMPOSITE_Z"].transform(
    lambda s: pd.qcut(
        s.rank(method="first"),
        q=5,
        labels=["G1", "G2", "G3", "G4", "G5"],
    )
)

ls = use[use["layer"].isin(["G1", "G5"])].copy()
layer_count = ls.groupby(["date", "layer"])["code"].transform("size")
ls["weight"] = np.where(
    ls["layer"].eq("G5"),
    1.0 / layer_count,
    -1.0 / layer_count,
)

weights_wide = (
    ls.pivot_table(
        index="date",
        columns="code",
        values="weight",
        aggfunc="sum",
    )
    .fillna(0.0)
    .sort_index()
)

returns_wide = (
    df.pivot_table(
        index="date",
        columns="code",
        values="next_ret",
        aggfunc="last",
    )
    .reindex(index=weights_wide.index, columns=weights_wide.columns)
)

held_missing_returns = int(
    (weights_wide.ne(0) & returns_wide.isna()).sum().sum()
)
if held_missing_returns:
    raise ValueError(
        f"持有仓位存在 {held_missing_returns} 个缺失收益"
    )

# 未持有股票的缺失收益不影响组合收益或权重漂移。
returns_for_drift = returns_wide.fillna(0.0)
portfolio_return = (weights_wide * returns_for_drift).sum(axis=1)

if not np.isfinite(portfolio_return).all():
    raise ValueError("组合收益存在非有限值")

drifted_weights = (
    weights_wide
    .mul(1 + returns_for_drift)
    .div(1 + portfolio_return, axis=0)
)

turnover_values = np.empty(len(weights_wide), dtype=float)
turnover_values[0] = (
    0.5 * np.abs(weights_wide.iloc[0].to_numpy()).sum()
)
turnover_values[1:] = (
    0.5
    * np.abs(
        weights_wide.iloc[1:].to_numpy()
        - drifted_weights.iloc[:-1].to_numpy()
    ).sum(axis=1)
)

turnover = pd.DataFrame({
    "date": weights_wide.index,
    "turnover": turnover_values,
    "is_initial": np.r_[True, np.zeros(len(weights_wide) - 1, dtype=bool)],
})
turnover["traded_notional"] = 2 * turnover["turnover"]

gross_long = (
    ls[ls["layer"] == "G5"].groupby("date")["next_ret"].mean()
)
gross_short = (
    ls[ls["layer"] == "G1"].groupby("date")["next_ret"].mean()
)
gross_returns = (gross_long - gross_short).sort_index()
weighted_returns = (weights_wide * returns_for_drift).sum(axis=1)
gross_consistency_error = float(
    (gross_returns - weighted_returns).abs().max()
)

saved_wide_path = processed_dir / "layer_returns_with_composite_wide.parquet"
saved_wide = pd.read_parquet(saved_wide_path)
saved_ls = (
    saved_wide.loc[saved_wide["factor"] == "COMPOSITE"]
    .set_index("date")["LS"]
    .sort_index()
)
sg2_consistency_error = float(
    (gross_returns - saved_ls.reindex(gross_returns.index)).abs().max()
)

if max(gross_consistency_error, sg2_consistency_error) > 1e-12:
    raise ValueError(
        "换手脚本毛收益与分层结果不一致: "
        f"{gross_consistency_error}, {sg2_consistency_error}"
    )

weights_long = (
    weights_wide.reset_index()
    .melt(id_vars="date", var_name="code", value_name="weight")
)
weights_long.to_parquet(
    processed_dir / "weights_composite_ls.parquet",
    index=False,
)

gross_returns.rename("gross_ret").reset_index().to_csv(
    processed_dir / "gross_composite_ls.csv",
    index=False,
    encoding="utf-8-sig",
)

turnover.to_csv(
    processed_dir / "turnover_composite_ls.csv",
    index=False,
    encoding="utf-8-sig",
)

steady_turnover = turnover.loc[~turnover["is_initial"], "turnover"]
quality = pd.DataFrame([{
    "weight_dates": len(weights_wide),
    "weight_codes": weights_wide.shape[1],
    "avg_holdings": float(weights_wide.ne(0).sum(axis=1).mean()),
    "held_missing_returns": held_missing_returns,
    "gross_exposure_mean": float(weights_wide.abs().sum(axis=1).mean()),
    "net_exposure_max_abs": float(weights_wide.sum(axis=1).abs().max()),
    "turnover_days": len(turnover),
    "turnover_nulls": int(turnover["turnover"].isna().sum()),
    "initial_turnover": float(turnover.loc[turnover["is_initial"], "turnover"].iloc[0]),
    "steady_turnover_mean": float(steady_turnover.mean()),
    "steady_turnover_median": float(steady_turnover.median()),
    "steady_turnover_max": float(steady_turnover.max()),
    "annualized_turnover": float(steady_turnover.mean() * 252),
    "gross_consistency_error": gross_consistency_error,
    "sg2_consistency_error": sg2_consistency_error,
}])
quality.to_csv(
    output_dir / "turnover_quality.csv",
    index=False,
    encoding="utf-8-sig",
)

print("换手率质量检查")
print(quality.round(8).to_string(index=False))
