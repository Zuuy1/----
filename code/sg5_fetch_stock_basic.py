from pathlib import Path

import baostock as bs
import pandas as pd


project_dir = Path(__file__).resolve().parents[1]
raw_dir = project_dir / "data" / "raw"
processed_dir = project_dir / "data" / "processed"
raw_dir.mkdir(parents=True, exist_ok=True)

output_path = raw_dir / "stock_basic.parquet"
if output_path.exists():
    print("已存在，跳过下载:", output_path)
    raise SystemExit(0)

codes = (
    pd.read_parquet(
        processed_dir / "factors_with_composite.parquet",
        columns=["code"],
    )["code"]
    .drop_duplicates()
    .sort_values()
    .tolist()
)

login = bs.login()
if login.error_code != "0":
    raise RuntimeError(f"Baostock 登录失败: {login.error_msg}")

rows = []
fields = None
try:
    for index, code in enumerate(codes, start=1):
        result = bs.query_stock_basic(code=code)
        if result.error_code != "0":
            print(f"[{index}/{len(codes)}] 失败 {code}: {result.error_msg}")
            continue

        if fields is None:
            fields = result.fields

        while result.next():
            rows.append(result.get_row_data())

        if index % 50 == 0:
            print(f"已查询 {index}/{len(codes)}")
finally:
    bs.logout()

basic = pd.DataFrame(rows, columns=fields)
if basic.empty:
    raise RuntimeError("上市基础信息为空")

basic = basic.drop_duplicates("code", keep="last")
if basic["code"].nunique() != len(codes):
    raise ValueError(
        f"上市信息不完整: {basic['code'].nunique()} / {len(codes)}"
    )

basic.to_parquet(output_path, index=False)
print("已保存:", output_path)
print("股票数量:", len(basic))
print(basic[["code", "code_name", "ipoDate", "type", "status"]].head())
