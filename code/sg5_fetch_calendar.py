from pathlib import Path

import baostock as bs
import pandas as pd


project_dir = Path(__file__).resolve().parents[1]
output_path = project_dir / "data" / "raw" / "trading_calendar.parquet"

if output_path.exists():
    print("已存在，跳过下载:", output_path)
    raise SystemExit(0)

login = bs.login()
if login.error_code != "0":
    raise RuntimeError(f"Baostock 登录失败: {login.error_msg}")

try:
    result = bs.query_history_k_data_plus(
        code="sh.000300",
        fields="date",
        start_date="2005-01-01",
        end_date="2025-12-31",
        frequency="d",
        adjustflag="3",
    )
    if result.error_code != "0":
        raise RuntimeError(result.error_msg)

    rows = []
    while result.next():
        rows.append(result.get_row_data())
finally:
    bs.logout()

calendar = pd.DataFrame(rows, columns=result.fields)
calendar["date"] = pd.to_datetime(calendar["date"])
calendar = (
    calendar.drop_duplicates("date")
    .sort_values("date")
    .reset_index(drop=True)
)

if calendar.empty or calendar["date"].isna().any():
    raise ValueError("交易日历为空或包含无效日期")

calendar.to_parquet(output_path, index=False)
print("已保存:", output_path)
print("交易日数量:", len(calendar))
print("范围:", calendar["date"].min(), "至", calendar["date"].max())
