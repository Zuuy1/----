import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


project_dir = Path(__file__).resolve().parents[1]
input_path = (
    project_dir / "data" / "processed" / "factors_standardized.parquet"
)
output_dir = project_dir / "output"
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(input_path)
df = df.sort_values(["code", "date"]).reset_index(drop=True)

# 下一交易日的交易状态
df["next_tradestatus"] = (
    df.groupby("code", group_keys=False)["tradestatus"].shift(-1)
)

factors = {
    "BP": "BP_Z",
    "MOM20": "MOM20_Z",
    "VOL60": "VOL60_Z",
}

records = []

for date, day in df.groupby("date"):
    record = {"date": date}

    for factor_name, factor_col in factors.items():
        sample = day.dropna(
            subset=[
                factor_col,
                "next_ret",
                "tradestatus",
                "next_tradestatus",
            ]
        )

        # 当前交易日和下一交易日都必须能够正常交易
        sample = sample[
            (sample["tradestatus"] == 1)
            & (sample["next_tradestatus"] == 1)
        ]

        if len(sample) < 30:
            record[f"{factor_name}_IC"] = np.nan
        else:
            record[f"{factor_name}_IC"] = sample[factor_col].corr(
                sample["next_ret"],
                method="spearman",
            )#算spearman相关系数，非参数方法，适合非正态分布的情况
        #corr方法计算列之间相关系数默认是pearson相关系数，适合正态分布的情况

    records.append(record)

ic_series = pd.DataFrame(records)

ic_series.to_csv(
    output_dir / "ic_series.csv",
    index=False,
    encoding="utf-8-sig",
)

summary_rows = []

for factor in ["BP", "MOM20", "VOL60"]:
    ic = ic_series[f"{factor}_IC"].dropna()

    summary_rows.append({
        "factor": factor,
        "days": len(ic),
        "IC_mean": ic.mean(),
        "IC_std": ic.std(ddof=1),
        "IR": ic.mean() / ic.std(ddof=1),
        "positive_rate": (ic > 0).mean(),
    })

summary = pd.DataFrame(summary_rows)

summary.to_csv(
    output_dir / "ic_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

print(summary.to_string(index=False))

'''ic_plot = ic_series.set_index("date")[
    ["BP_IC", "MOM20_IC", "VOL60_IC"]
]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

for ax, column, title in zip(
    axes,
    ic_plot.columns,
    ["BP Rank IC", "MOM20 Rank IC", "VOL60 Rank IC"],
):
    ax.plot(ic_plot.index, ic_plot[column], linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.grid(alpha=0.25)

plt.tight_layout()
plt.savefig(output_dir / "ic_series.png", dpi=160)
plt.show()'''
