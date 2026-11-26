"""Fetch annual profit data with publication dates from Baostock."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import baostock as bs
import pandas as pd


YEARS = range(2019, 2026)
MAX_WORKERS = 6


def _login_worker():
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"Baostock 登录失败: {login.error_msg}")


def _fetch_code(code, raw_dir_string):
    target = Path(raw_dir_string) / f"{code.replace('.', '_')}.parquet"
    if target.exists():
        return code, "cached", len(pd.read_parquet(target))

    frames = []
    for year in YEARS:
        result = bs.query_profit_data(
            code=code,
            year=year,
            quarter=4,
        )
        if result.error_code != "0":
            continue
        rows = []
        while result.next():
            rows.append(result.get_row_data())
        if rows:
            frames.append(
                pd.DataFrame(rows, columns=result.fields)
            )

    if not frames:
        return code, "empty", 0

    data = pd.concat(frames, ignore_index=True)
    data = data.drop_duplicates("statDate", keep="last")
    data.to_parquet(target, index=False)
    return code, "saved", len(data)


def main():
    project_dir = Path(__file__).resolve().parents[2]
    raw_dir = project_dir / "data" / "raw" / "fundamentals_annual"
    raw_dir.mkdir(parents=True, exist_ok=True)

    codes = (
        pd.read_parquet(
            project_dir
            / "data"
            / "processed"
            / "factors_with_composite.parquet",
            columns=["code"],
        )["code"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    missing = [
        code
        for code in codes
        if not (raw_dir / f"{code.replace('.', '_')}.parquet").exists()
    ]

    completed = 0
    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        initializer=_login_worker,
    ) as pool:
        futures = {
            pool.submit(_fetch_code, code, str(raw_dir)): code
            for code in missing
        }
        for future in as_completed(futures):
            code, status, rows = future.result()
            if status == "empty":
                print("无年度数据:", code)
            completed += 1
            if completed % 25 == 0:
                print(f"已处理 {completed}/{len(missing)}")

    files = sorted(raw_dir.glob("*.parquet"))
    if not files:
        raise RuntimeError("没有取得任何年度基本面数据")
    combined = pd.concat(
        [pd.read_parquet(file) for file in files],
        ignore_index=True,
    )
    combined = (
        combined.drop_duplicates(["code", "statDate"], keep="last")
        .sort_values(["code", "statDate"])
        .reset_index(drop=True)
    )
    combined["pubDate"] = pd.to_datetime(
        combined["pubDate"],
        errors="coerce",
    )
    combined["statDate"] = pd.to_datetime(
        combined["statDate"],
        errors="coerce",
    )
    if combined["pubDate"].isna().any():
        raise ValueError("基本面数据存在无效 pubDate")

    output_path = project_dir / "data" / "raw" / "fundamentals_annual.parquet"
    combined.to_parquet(output_path, index=False)
    print("已保存:", output_path)
    print(
        "股票数:", combined["code"].nunique(),
        "记录数:", len(combined),
        "公告日期范围:", combined["pubDate"].min(),
        "至", combined["pubDate"].max(),
    )


if __name__ == "__main__":
    main()
