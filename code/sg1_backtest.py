import pandas as pd
from pathlib import Path


# 1. 路径
project_dir = Path(__file__).resolve().parent
input_path = (
    project_dir / "data" / "processed" / "factors_standardized.parquet"
)
processed_dir = project_dir / "data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

flat_path = processed_dir / "layer_returns.parquet"
wide_path = processed_dir / "layer_returns_wide.parquet"


# 2. 读取标准化后的因子面板
df = pd.read_parquet(input_path)
df = df.sort_values(["code", "date"]).reset_index(drop=True)

# 获取同一只股票下一交易日的交易状态
df["next_tradestatus"] = (
    df.groupby("code", group_keys=False)["tradestatus"].shift(-1)
)

factors = {
    "BP": "BP_Z",
    "MOM20": "MOM20_Z",
    "VOL60": "VOL60_Z",
}


# 3. 每个交易日、每个因子分别分成 G1-G5
layer_records = []

for date, day in df.groupby("date", sort=True):
    for factor_name, factor_col in factors.items():
        sample = day.dropna(
            subset=[
                factor_col,
                "next_ret",
                "tradestatus",
                "next_tradestatus",
            ]
        ).copy()

        # 当前和下一交易日都必须能够正常交易
        sample = sample[
            (sample["tradestatus"] == 1)
            & (sample["next_tradestatus"] == 1)
        ]

        # 股票太少时无法形成稳定的五组
        if len(sample) < 25:
            continue

        # 先排名再 qcut，避免相同因子值导致分组边界异常
        sample["layer"] = pd.qcut(
            sample[factor_col].rank(method="first"),
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


# 4. 保存长表
layer_returns = pd.DataFrame(layer_records)

layer_returns.to_parquet(flat_path, index=False)


# 5. 转成宽表，并计算 G5-G1
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

layer_returns_wide = layer_returns_wide.sort_values(
    ["factor", "date"]
).reset_index(drop=True)

layer_returns_wide.to_parquet(wide_path, index=False)


# 6. 输出基本检查
print("长表行数:", len(layer_returns))
print("宽表行数:", len(layer_returns_wide))
print("因子数量:", layer_returns["factor"].nunique())
print("日期范围:", layer_returns["date"].min(), "至", layer_returns["date"].max())

print("\n每日每组股票数量:")
print(
    layer_returns["n_stocks"]
    .describe()
    .to_string()
)

print("\n宽表前五行:")
print(layer_returns_wide.head().to_string(index=False))

