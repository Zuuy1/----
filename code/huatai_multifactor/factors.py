"""Huatai-style daily price-volume and annual fundamental factors."""

from pathlib import Path

import numpy as np
import pandas as pd


PRICE_FACTORS = [
    "MOM20",
    "MOM60",
    "MOM120",
    "REV5",
    "VOL20",
    "VOL60",
    "TURN20",
    "ILLIQ20",
    "HIGH52",
]

FUNDAMENTAL_FACTORS = [
    "EP",
    "BP",
    "SP",
    "CFP",
    "ROE",
    "NPM",
    "GPM",
    "PROFIT_YOY",
    "REVENUE_YOY",
]


def _group_change(series, code, periods):
    return series.groupby(code, sort=False).pct_change(
        periods,
        fill_method=None,
    )


def _rolling_by_code(df, series, window, operation):
    values = series.groupby(df["code"], sort=False).transform(
        lambda s: getattr(
            s.rolling(window, min_periods=window),
            operation,
        )()
    )
    return values


def _attach_fundamentals(df, project_dir):
    project_dir = Path(project_dir)
    fundamentals = pd.read_parquet(
        project_dir / "data" / "raw" / "fundamentals_annual.parquet"
    )
    numeric_columns = [
        "roeAvg",
        "npMargin",
        "gpMargin",
        "netProfit",
        "MBRevenue",
    ]
    for column in numeric_columns:
        fundamentals[column] = pd.to_numeric(
            fundamentals[column],
            errors="coerce",
        )
    fundamentals["pubDate"] = pd.to_datetime(
        fundamentals["pubDate"],
        errors="coerce",
    )
    fundamentals["statDate"] = pd.to_datetime(
        fundamentals["statDate"],
        errors="coerce",
    )
    fundamentals["year"] = fundamentals["statDate"].dt.year

    previous = fundamentals[
        ["code", "year", "netProfit", "MBRevenue"]
    ].copy()
    previous["year"] += 1
    previous = previous.rename(columns={
        "netProfit": "prev_netProfit",
        "MBRevenue": "prev_MBRevenue",
    })
    fundamentals = fundamentals.merge(
        previous,
        on=["code", "year"],
        how="left",
        validate="many_to_one",
    )
    fundamentals["PROFIT_YOY"] = (
        (fundamentals["netProfit"] - fundamentals["prev_netProfit"])
        / fundamentals["prev_netProfit"].abs().where(
            fundamentals["prev_netProfit"].ne(0)
        )
    )
    fundamentals["REVENUE_YOY"] = (
        (fundamentals["MBRevenue"] - fundamentals["prev_MBRevenue"])
        / fundamentals["prev_MBRevenue"].abs().where(
            fundamentals["prev_MBRevenue"].ne(0)
        )
    )
    fundamentals["ROE"] = fundamentals["roeAvg"]
    fundamentals["NPM"] = fundamentals["npMargin"]
    fundamentals["GPM"] = fundamentals["gpMargin"]

    calendar = (
        pd.read_parquet(
            project_dir / "data" / "raw" / "trading_calendar.parquet"
        )["date"]
        .drop_duplicates()
        .sort_values()
        .to_numpy(dtype="datetime64[ns]")
    )
    next_positions = np.searchsorted(
        calendar,
        fundamentals["pubDate"].to_numpy(dtype="datetime64[ns]"),
        side="right",
    )
    valid_position = (
        fundamentals["pubDate"].notna()
        & (next_positions < len(calendar))
    )
    fundamentals["effective_date"] = pd.NaT
    fundamentals.loc[valid_position, "effective_date"] = calendar[
        next_positions[valid_position.to_numpy()]
    ]
    fundamentals = fundamentals.dropna(
        subset=["effective_date"]
    ).sort_values(["code", "effective_date", "statDate"])

    fundamental_columns = [
        "ROE",
        "NPM",
        "GPM",
        "PROFIT_YOY",
        "REVENUE_YOY",
    ]
    result = pd.DataFrame(index=df.index, columns=fundamental_columns)

    for code, daily_group in df.groupby("code", sort=False):
        report_group = fundamentals[
            fundamentals["code"] == code
        ].sort_values(["effective_date", "statDate"])
        if report_group.empty:
            continue
        report_group = report_group.drop_duplicates(
            "effective_date",
            keep="last",
        )
        daily_for_merge = daily_group.sort_values("date").copy()
        daily_for_merge["date"] = daily_for_merge["date"].astype(
            "datetime64[ns]"
        )
        report_for_merge = report_group[
            ["effective_date", *fundamental_columns]
        ].copy()
        report_for_merge["effective_date"] = (
            report_for_merge["effective_date"].astype(
                "datetime64[ns]"
            )
        )
        merged = pd.merge_asof(
            daily_for_merge,
            report_for_merge,
            left_on="date",
            right_on="effective_date",
            direction="backward",
            allow_exact_matches=True,
        )
        merged.index = daily_for_merge.index
        result.loc[merged.index, fundamental_columns] = (
            merged[fundamental_columns].to_numpy()
        )

    return df.join(result)


def build_huatai_factors(df, project_dir):
    data = df.sort_values(["code", "date"]).reset_index(drop=True).copy()
    data["ret"] = data["close"] / data["preclose"] - 1
    data["MOM20"] = _group_change(data["close"], data["code"], 20)
    data["MOM60"] = _group_change(data["close"], data["code"], 60)
    data["MOM120"] = _group_change(data["close"], data["code"], 120)
    data["REV5"] = -_group_change(data["close"], data["code"], 5)
    data["VOL20"] = -_rolling_by_code(
        data,
        data["ret"],
        20,
        "std",
    )
    data["VOL60"] = -_rolling_by_code(
        data,
        data["ret"],
        60,
        "std",
    )
    data["TURN20"] = -_rolling_by_code(
        data,
        data["turn"],
        20,
        "mean",
    )
    data["ILLIQ20"] = _rolling_by_code(
        data,
        data["ret"].abs() / data["amount"].where(data["amount"] > 0),
        20,
        "mean",
    )
    high_250 = _rolling_by_code(data, data["close"], 250, "max")
    data["HIGH52"] = data["close"] / high_250

    data["EP"] = 1 / data["peTTM"].where(data["peTTM"] > 0)
    data["BP"] = 1 / data["pbMRQ"].where(data["pbMRQ"] > 0)
    data["SP"] = 1 / data["psTTM"].where(data["psTTM"] > 0)
    data["CFP"] = 1 / data["pcfNcfTTM"].where(data["pcfNcfTTM"] > 0)
    data = _attach_fundamentals(data, project_dir)

    raw_columns = PRICE_FACTORS + FUNDAMENTAL_FACTORS
    for column in raw_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        data[column] = data[column].replace(
            [np.inf, -np.inf],
            np.nan,
        )

    z_columns = []
    for column in raw_columns:
        z_column = f"{column}_Z"
        data[z_column] = data.groupby(
            "date",
            group_keys=False,
        )[column].transform(_robust_zscore)
        z_columns.append(z_column)

    price_z = [f"{column}_Z" for column in PRICE_FACTORS]
    fund_z = [f"{column}_Z" for column in FUNDAMENTAL_FACTORS]

    data["PRICE_COMPOSITE_Z"] = data[price_z].mean(
        axis=1,
    ).where(data[price_z].notna().sum(axis=1) >= 5)
    data["FUNDAMENTAL_COMPOSITE_Z"] = data[fund_z].mean(
        axis=1,
    ).where(data[fund_z].notna().sum(axis=1) >= 5)
    data["TOTAL_COMPOSITE_Z"] = (
        data["PRICE_COMPOSITE_Z"]
        + data["FUNDAMENTAL_COMPOSITE_Z"]
    ) / 2
    return data


def _robust_zscore(series, n_mad=5, min_obs=30):
    valid = series.dropna()
    if len(valid) < min_obs:
        return pd.Series(np.nan, index=series.index, dtype=float)
    median = valid.median()
    mad = (valid - median).abs().median()
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    lower = median - n_mad * 1.4826 * mad
    upper = median + n_mad * 1.4826 * mad
    clipped = valid.clip(lower=lower, upper=upper)
    std = clipped.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    output = pd.Series(np.nan, index=series.index, dtype=float)
    output.loc[valid.index] = (
        clipped - clipped.mean()
    ) / std
    return output


def factor_columns():
    return (
        PRICE_FACTORS
        + FUNDAMENTAL_FACTORS
        + [
            "PRICE_COMPOSITE_Z",
            "FUNDAMENTAL_COMPOSITE_Z",
            "TOTAL_COMPOSITE_Z",
        ]
    )
