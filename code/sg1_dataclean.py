import pandas as pd
from pathlib import Path



#清洗
project_dir = Path(__file__).resolve().parent
processed_dir = project_dir /"data"/"processed"
processed_dir.mkdir(parents=True, exist_ok=True)
raw_dir = project_dir / "data" / "raw" / "prices"

files=sorted(raw_dir.glob("*.parquet"))
#raw_dir.glob("*.parquet")：在 raw_dir 下按通配符找文件，产出的是 Path 对象
#sorted(...)：把迭代器整个消费掉，变成一个排好序的 list。
frames=[pd.read_parquet(file) for file in files]  #列表推导式，与分开写等价
df=pd.concat(frames,ignore_index=True)  #拼成一张大表,且让index不重叠

bad_adjust = df[df["adjustflag"] != "2"].copy()
print(
    bad_adjust
    .groupby("code")
    .agg(
        bad_rows=("date", "size"),
        first_date=("date", "min"),
        last_date=("date", "max")
    )
)
df = df[df["adjustflag"] == "2"].copy()
#处理部分坏数据只保留前复权（"2"）的行，覆盖原表

df["date"]=pd.to_datetime(df["date"])
numeric_cols=["open", "high", "low", "close", "preclose",
    "volume", "amount", "turn", "pctChg",
    "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"]
for col in numeric_cols:
    df[col]=pd.to_numeric(df[col],errors="coerce")

df["tradestatus"] = pd.to_numeric(
    df["tradestatus"], errors="coerce"
).astype("Int64")

df["isST"] = pd.to_numeric(
    df["isST"], errors="coerce"
).astype("Int64")

df=df.drop_duplicates(["date", "code"], keep="last")
#把(日期+股票代码)​当作唯一键，同一组合只留一行。
#把多个 parquet 拼起来之后，同一只股票同一天很可能出现多次（重复抓取、两次跑脚本、文件名重叠）
df=df.sort_values(["code", "date"])#先按股票代码分组，组内按日期升序
df=df.reset_index(drop=True)#重置索引，drop=True清除旧索引
#把数据去重 → 排序 → 重置索引串起来

print("股票数量:", df["code"].nunique())
print("总行数:", len(df))
print("最早日期:", df["date"].min())
print("最晚日期:", df["date"].max())
print("重复记录:", df.duplicated(["date", "code"]).sum())
print("非正收盘价:", (df["close"] <= 0).sum())
print("复权标记:", df["adjustflag"].unique())
print("停牌记录:", (df["tradestatus"] == 0).sum())
print("pbMRQ 缺失率:", df["pbMRQ"].isna().mean())

df.to_parquet(
    processed_dir / "hs300_daily.parquet",
    index=False
)

quality = pd.DataFrame([
    {"指标": "股票数量", "值": df["code"].nunique()},
    {"指标": "总行数", "值": len(df)},
    {"指标": "最早日期", "值": str(df["date"].min().date())},
    {"指标": "最晚日期", "值": str(df["date"].max().date())},
    {"指标": "重复记录", "值": df.duplicated(["date", "code"]).sum()},
    {"指标": "停牌记录", "值": (df["tradestatus"] == 0).sum()},
    {"指标": "PB缺失率", "值": df["pbMRQ"].isna().mean()},
])

quality.to_csv(
    processed_dir / "data_quality.csv",
    index=False,
    encoding="utf-8-sig"
)

print("已保存")