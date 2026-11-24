import pandas as pd
import numpy as np
from pathlib import Path

project_dir=Path(__file__).resolve().parents[1]
input_path=project_dir/'data'/'processed'/'hs300_daily.parquet'
output_path=project_dir/'data'/'processed'/"factors_daily.parquet"

df=pd.read_parquet(input_path)
df=df.sort_values(["code","date"]).reset_index(drop=True)

grouped = df.groupby("code",group_keys=False)

df["ret"] = grouped["close"].pct_change()
#pct_change计算相邻两行的百分比变化（增长率）
df["MOM20"] = grouped["close"].transform(
    lambda s: s.pct_change(20, fill_method=None)
)
#transform：**分组计算，返回和原 df 一样行数的结果，计算结果直接贴回原表
df["VOL60"] = df.groupby("code")["ret"].transform(
    lambda s: s.rolling(60, min_periods=60).std(ddof=0)
)
#一个固定大小的窗口，沿着序列一步步向右滑动，每次只拿窗口内的数据做计算
df["BP"] = 1 / df["pbMRQ"].where(df["pbMRQ"] > 0)
df["next_ret"] = (
    grouped["close"].shift(-1)
    .div(df["close"])
    .sub(1)
)
#取下一行 → 除以今天 → 减 1
#BP     = 1 / pbMRQ
#MOM20  = 过去 20 个交易日收益率
#VOL60  = 过去 60 个交易日收益率标准差
cols = ["BP", "MOM20", "VOL60", "next_ret"]
df[cols] = df[cols].replace([np.inf, -np.inf], np.nan)


df.to_parquet(output_path, index=False)
