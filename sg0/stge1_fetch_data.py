import sys
import os
print("python版本:",sys.version)
import baostock as bs
import pandas as pd
import matplotlib.pyplot as plt

print("库导入")

#登录
lg=bs.login()
print("login respond error_code:",lg.error_code)
print("login respond error_msg:",lg.error_msg)

#获取全部股票列表
# rs=bs.query_all_stock(day="2026-07-24")
# data_list=[]
# while (rs.error_code == "0") and rs.next():
#     data_list.append(rs.get_row_data())

# stock_df = pd.DataFrame(data_list,columns=rs.fields)
# print("股票数量：",len(stock_df))
# stock_df.head(10)

#获取单只股票k线
code = "sh.600000"
fields="date,open,high,low,close,volume,amount,turn,pctChg,peTTM"
rs = bs.query_history_k_data_plus(
    code=code,
    fields=fields,
    start_date="2023-01-01",
    end_date="2023-12-31",
    frequency="d",
    adjustflag="2"#做对比时要统一成3吗？
)
data_list=[]
assert rs is not None
while (rs.error_code == "0") and rs.next():
    data_list.append(rs.get_row_data())

k_df=pd.DataFrame(data_list,columns=rs.fields)
k_df["date"]=pd.to_datetime(k_df["date"])
num_cols=[
    "open","high","low","close","volume","amount","turn","pctChg","peTTM"]
for c in num_cols:
    k_df[c]=pd.to_numeric(k_df[c],errors="coerce")           #字符串转为数字类型
    #pd.to_numeric(arg, errors='raise'(报错)'coerce'(NaN), downcast=None)
print(k_df.dtypes)
print("缺失值统计:",'\n',k_df.isna().sum())
k_df.describe()
plt.rcParams["font.sans-serif"] = ["SimHei","Microsoft YaHei","Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] =False

plt.figure(figsize=(12,5))
plt.plot(k_df["date"], k_df["close"], label=f"{code}收盘价")
plt.title("2023年收盘价走势")
plt.xlabel("日期")
plt.ylabel("收盘价")
plt.grid(alpha=0.3)
plt.legend()
plt.show()

index_rs=bs.query_history_k_data_plus(
    code="sh.000001",
    fields="date,close",
    start_date="2023-01-01",
    end_date="2023-12-31",
    frequency="d",
    adjustflag="3")

assert index_rs is not None
index_list=[]
while(index_rs.error_code=="0") and index_rs.next():
    index_list.append(index_rs.get_row_data())

index_df=pd.DataFrame(index_list,columns=index_rs.fields)
index_df["date"]=pd.to_datetime(index_df["date"])
index_df["close"]=pd.to_numeric(index_df["close"],errors="coerce")

compare_df=pd.merge(k_df[["date","close"]].rename(columns={"close":"stock_close"})
                    ,index_df[["date","close"]].rename(columns={"close":"index_close"})
                    ,on="date",how="inner")
#pd.merge按key合并两个DataFrame
#pd.merge(left, right, on=None, how='inner',
#  left_on=None, right_on=None, suffixes=('_x','_y'))
# `on`：两张表**同名的关联列**，比如 `on='id'`
# `left_on` / `right_on`：两张表关联列名字不一样的时候用。比如左表是`user_id`，右表是`uid`
# `how`：合并方式，四种最常用：
# `inner`（默认，内连接）：只保留两边都匹配上的行（交集）
# `left`（左连接）：保留左表全部行，右表匹配不上的填 NaN
# `outer`（全外连接）：两边所有行都保留（并集）
# `suffixes`：如果两个表有**同名但不是关联键**的列，自动加后缀区分，默认 `_x`（左）、`_y`（右
#rename给index或columns改名  df_new = df.rename(columns={"a":"id", "b":"value"})

compare_df["stock_return"]=(compare_df["stock_close"]/compare_df["stock_close"].iloc[0]-1)
compare_df["index_return"]=(compare_df["index_close"]/compare_df["index_close"].iloc[0]-1)
#计算回报率

plt.figure(figsize=(12,5))
plt.plot(compare_df["date"], compare_df["stock_return"], label="个股累计收益率")
plt.plot(compare_df["date"], compare_df["index_return"], label="上证指数累计收益率")
plt.title("个股与大盘累计收益率对比(2023)")
plt.xlabel("日期")
plt.ylabel("累计收益率")
plt.grid(alpha=0.3)
plt.legend()
plt.show()

profit_rs=bs.query_profit_data(code="sh.600000",year=2023,quarter=1)
profit_list=[]
while(profit_rs.error_code=="0")and profit_rs.next():
    profit_list.append(profit_rs.get_row_data())

profit_df=pd.DataFrame(profit_list,columns=profit_rs.fields)
print(profit_df)

lg=bs.logout()

save_dir="output"
os.makedirs(save_dir,exist_ok=True)
k_path=os.path.join(save_dir,"sh600000_kdata_2023.csv")
c_path=os.path.join(save_dir,"compare_stock_index_2023.csv")
p_path=os.path.join(save_dir,"sh600000_profit_2023q1.csv")
# 拼接：data文件夹 + result.csv
#save_path = os.path.join("data", "result.csv")
# 拼接：data文件夹 + result.csv
#save_path = os.path.join("data", "result.csv")

k_df.to_csv(k_path,index=False,encoding="utf-8-sig")
compare_df.to_csv(c_path, index=False, encoding="utf-8-sig")
profit_df.to_csv(p_path, index=False, encoding="utf-8-sig")

print(k_path)
print(c_path)
print(p_path)