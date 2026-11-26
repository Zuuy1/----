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

cost_levels_bps = [0, 5, 10, 15, 20, 30, 50]

gross = pd.read_csv(
    processed_dir / "gross_composite_ls.csv",
    parse_dates=["date"],
)
turnover = pd.read_csv(
    processed_dir / "turnover_composite_ls.csv",
    parse_dates=["date"],
)

data = gross.merge(
    turnover[["date", "turnover", "traded_notional", "is_initial"]],
    on="date",
    how="inner",
    validate="one_to_one",
).sort_values("date").reset_index(drop=True)

if len(data) != len(gross) or len(data) != len(turnover):
    raise ValueError("毛收益与换手率日期未完整对齐")
if data[["gross_ret", "turnover", "traded_notional"]].isna().any().any():
    raise ValueError("成本和收益数据存在缺失")
if (data["traded_notional"] < 0).any():
    raise ValueError("成交额不能为负数")

summary_rows = []
curve_rows = []

for cost_bps in cost_levels_bps:
    daily_cost = data["traded_notional"] * cost_bps / 1e4
    net_ret = data["gross_ret"] - daily_cost
    nav = (1 + net_ret).cumprod()
    drawdown = nav / nav.cummax() - 1

    summary_rows.append({
        "cost_bps_oneway": cost_bps,
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
        "total_cost": daily_cost.sum(),
        "avg_daily_cost_bp": daily_cost.mean() * 1e4,
    })

    curve_rows.append(pd.DataFrame({
        "date": data["date"],
        "cost_bps_oneway": cost_bps,
        "gross_ret": data["gross_ret"],
        "daily_cost": daily_cost,
        "net_ret": net_ret,
        "nav": nav,
    }))

summary = pd.DataFrame(summary_rows)
curves = pd.concat(curve_rows, ignore_index=True)

summary.to_csv(
    output_dir / "cost_sensitivity_composite_ls.csv",
    index=False,
    encoding="utf-8-sig",
)
curves.to_parquet(
    processed_dir / "net_nav_composite_ls.parquet",
    index=False,
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(12, 6))

for cost_bps, group in curves.groupby("cost_bps_oneway", sort=True):
    ax.plot(
        group["date"],
        group["nav"],
        linewidth=1,
        label=f"{cost_bps} bp",
    )

ax.set_title("COMPOSITE G5-G1 成本敏感性")
ax.set_xlabel("日期")
ax.set_ylabel("累计净值")
ax.grid(alpha=0.25)
ax.legend(ncol=2)
plt.tight_layout()
plt.savefig(
    output_dir / "cost_sensitivity_composite_ls.png",
    dpi=160,
    bbox_inches="tight",
)
plt.close()

print("日期范围:", data["date"].min(), "至", data["date"].max())
print("首日建仓成交额:", float(data.loc[data["is_initial"], "traded_notional"].iloc[0]))
print(summary.round(4).to_string(index=False))
