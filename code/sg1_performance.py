import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# 1. 路径
project_dir = Path(__file__).resolve().parent
input_path = (
    project_dir / "data" / "processed" / "layer_returns_wide.parquet"
)
output_dir = project_dir / "output"
output_dir.mkdir(parents=True, exist_ok=True)

# 2. 读取 G1-G5 每日收益
layer_returns = pd.read_parquet(input_path)
layer_returns["date"] = pd.to_datetime(layer_returns["date"])

portfolio_names = ["G1", "G2", "G3", "G4", "G5", "LS"]
performance_rows = []
nav_records = []


# 3. 分别计算每个因子、每个组合的绩效
for factor in ["BP", "MOM20", "VOL60"]:
    factor_df = (
        layer_returns
        .query("factor == @factor")
        .sort_values("date")
        .set_index("date")
    )

    for portfolio in portfolio_names:
        daily_ret = pd.to_numeric(
            factor_df[portfolio],
            errors="coerce"
        ).dropna()

        if daily_ret.empty:
            continue

        nav = (1 + daily_ret).cumprod()#把每天的收益率连乘起来，1 块钱滚成多少。
        cumulative_return = nav.iloc[-1] - 1#六年总共赚多少

        annual_return = (
            nav.iloc[-1] ** (252 / len(daily_ret)) - 1
        )#年化（几何平均），252 = 一年交易日
        annual_vol = daily_ret.std(ddof=1) * np.sqrt(252)#日收益率样本标准差->年化波动率，越大越颠

        if daily_ret.std(ddof=1) > 0:
            sharpe = (
                daily_ret.mean()
                / daily_ret.std(ddof=1)
                * np.sqrt(252)
            )#夏普比率 每单位风险换多少收益 日收益率均值/日样本标准差
        else:
            sharpe = np.nan

        drawdown = nav / nav.cummax() - 1#最惨从高点跌多少

        performance_rows.append({
            "factor": factor,
            "portfolio": portfolio,
            "days": len(daily_ret),
            "cumulative_return": cumulative_return,
            "annual_return": annual_return,
            "annual_vol": annual_vol,
            "sharpe": sharpe,
            "max_drawdown": drawdown.min(),
            "positive_rate": (daily_ret > 0).mean(),
        })

        nav_records.append(
            pd.DataFrame({
                "date": nav.index,
                "factor": factor,
                "portfolio": portfolio,
                "nav": nav.values,
            })
        )


# 4. 保存绩效表
performance = pd.DataFrame(performance_rows)

performance.to_csv(
    output_dir / "performance_table.csv",
    index=False,
    encoding="utf-8-sig",
)

print("绩效表:")
print(performance.round(4).to_string(index=False))


# 5. 绘制每个因子的累计净值
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(
    3,
    1,
    figsize=(13, 11),
    sharex=True,
)

for ax, factor in zip(axes, ["BP", "MOM20", "VOL60"]):
    factor_df = layer_returns.query("factor == @factor").set_index("date")

    for portfolio in ["G1", "G2", "G3", "G4", "G5"]:
        daily_ret = pd.to_numeric(
            factor_df[portfolio],
            errors="coerce"
        ).dropna()
        nav = (1 + daily_ret).cumprod()
        ax.plot(
            nav.index,
            nav,
            linewidth=1,
            label=portfolio,
        )

    ls_daily_ret = pd.to_numeric(
        factor_df["LS"],
        errors="coerce"
    ).dropna()
    ls_nav = (1 + ls_daily_ret).cumprod()

    ax.plot(
        ls_nav.index,
        ls_nav,
        color="black",
        linewidth=1.5,
        linestyle="--",
        label="G5-G1",
    )

    ax.set_title(f"{factor}：分层累计净值")
    ax.set_ylabel("累计净值")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3)

axes[-1].set_xlabel("日期")

plt.tight_layout()
plt.savefig(
    output_dir / "layered_returns.png",
    dpi=160,
    bbox_inches="tight",
)
plt.show()