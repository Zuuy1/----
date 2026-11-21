import numpy as np
import pandas as pd
from pathlib import Path

project_dir = Path(__file__).resolve().parent
input_path = project_dir / "data" / "processed" / "factors_daily.parquet"
output_path = project_dir / "data" / "processed" / "factors_standardized.parquet"

def robust_zscore(series, n_mad=5, min_obs=30):
    valid = series.dropna()
    #把缺失值（NaN）扔掉，只留真实有值的
    if len(valid) < min_obs:
        return pd.Series(np.nan, index=series.index, dtype=float)

    median = valid.median()#排序后取正中间那个数
    mad = (valid - median).abs().median()#中位数绝对偏差

    if not np.isfinite(mad) or mad == 0:
        return pd.Series(np.nan, index=series.index, dtype=float)

    lower = median - n_mad * 1.4826 * mad#正态假设下的换算系数
    upper = median + n_mad * 1.4826 * mad

    clipped = valid.clip(lower=lower, upper=upper)#把超出区间的数压到边界上，中间的保持不动。
    mean = clipped.mean()#平均值
    std = clipped.std(ddof=0)#标准差（ddof=0表示除以 n）
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.nan, index=series.index, dtype=float)

    result = pd.Series(np.nan, index=series.index, dtype=float)
    result.loc[valid.index] = (clipped - mean) / std
    return result

df = pd.read_parquet(input_path)
df = df.sort_values(["date", "code"]).reset_index(drop=True)

for factor in ["BP", "MOM20", "VOL60"]:
    df[f"{factor}_Z"] = (
        df.groupby("date", group_keys=False)[factor]
        .transform(robust_zscore)#对每一块调用你的函数，把返回结果按原索引拼回原表
    )

df.to_parquet(output_path, index=False)

