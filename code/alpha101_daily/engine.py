"""Execution engine shared by the daily Alpha101 pilot factors."""

from pathlib import Path

import numpy as np
import pandas as pd


MIN_NAMES = 50
MIN_LISTING_DAYS = 250
COMMISSION_BP = 2.5
TRANSFER_BP = 0.1
STAMP_CUT_DATE = pd.Timestamp("2023-08-28")


def prepare_execution_data(df, project_dir):
    project_dir = Path(project_dir)
    data = df.sort_values(["code", "date"]).reset_index(drop=True).copy()

    stock_basic = pd.read_parquet(
        project_dir / "data" / "raw" / "stock_basic.parquet",
        columns=["code", "ipoDate"],
    )
    data = data.merge(
        stock_basic,
        on="code",
        how="left",
        validate="many_to_one",
    )
    data["ipoDate"] = pd.to_datetime(data["ipoDate"], errors="coerce")
    data = data.sort_values(["code", "date"]).reset_index(drop=True)

    grouped = data.groupby("code", group_keys=False)
    shift_fields = [
        "date",
        "open",
        "high",
        "low",
        "pctChg",
        "tradestatus",
        "isST",
    ]
    for field in shift_fields:
        data[f"entry_{field}"] = grouped[field].shift(-1)
        data[f"exit_{field}"] = grouped[field].shift(-2)

    calendar = (
        pd.read_parquet(
            project_dir / "data" / "raw" / "trading_calendar.parquet"
        )["date"]
        .drop_duplicates()
        .sort_values()
        .to_numpy(dtype="datetime64[ns]")
    )
    valid_dates = data["ipoDate"].notna() & data["entry_date"].notna()
    listing_days = pd.Series(np.nan, index=data.index, dtype=float)
    entry_position = np.searchsorted(
        calendar,
        data.loc[valid_dates, "entry_date"].to_numpy(
            dtype="datetime64[ns]"
        ),
        side="left",
    )
    ipo_position = np.searchsorted(
        calendar,
        data.loc[valid_dates, "ipoDate"].to_numpy(
            dtype="datetime64[ns]"
        ),
        side="left",
    )
    listing_days.loc[valid_dates] = entry_position - ipo_position + 1
    data["listing_trading_days"] = listing_days

    entry_limit = _limit_threshold(
        data["code"],
        data["entry_date"],
        data["entry_isST"],
    )
    exit_limit = _limit_threshold(
        data["code"],
        data["exit_date"],
        data["exit_isST"],
    )
    lock_flags = [
        _locked_at_open(
            data["entry_open"],
            data["entry_high"],
            data["entry_pctChg"],
            entry_limit,
            upper=True,
        ),
        _locked_at_open(
            data["entry_open"],
            data["entry_low"],
            data["entry_pctChg"],
            entry_limit,
            upper=False,
        ),
        _locked_at_open(
            data["exit_open"],
            data["exit_high"],
            data["exit_pctChg"],
            exit_limit,
            upper=True,
        ),
        _locked_at_open(
            data["exit_open"],
            data["exit_low"],
            data["exit_pctChg"],
            exit_limit,
            upper=False,
        ),
    ]
    locked = pd.concat(lock_flags, axis=1).fillna(False).any(axis=1)

    data["exec_ret"] = data["exit_open"] / data["entry_open"] - 1
    data["base_eligible"] = (
        data["tradestatus"].eq(1)
        & data["entry_date"].notna()
        & data["exit_date"].notna()
        & (data["entry_open"] > 0)
        & (data["exit_open"] > 0)
        & data["entry_tradestatus"].eq(1)
        & data["exit_tradestatus"].eq(1)
        & data["isST"].eq(0)
        & data["entry_isST"].eq(0)
        & data["exit_isST"].eq(0)
        & (data["listing_trading_days"] >= MIN_LISTING_DAYS)
        & ~locked
    ).fillna(False)

    if data.loc[data["base_eligible"], "exec_ret"].isna().any():
        raise ValueError("可执行样本存在缺失收益")
    return data


def _limit_threshold(codes, dates, is_st):
    star = codes.str.startswith("sh.688")
    chinext = codes.str.startswith("sz.30")
    after_reform = dates >= pd.Timestamp("2020-08-24")
    return np.select(
        [
            is_st.eq(1).fillna(False).to_numpy(dtype=bool),
            star.fillna(False).to_numpy(dtype=bool),
            (chinext & after_reform)
            .fillna(False)
            .to_numpy(dtype=bool),
        ],
        [5.0, 20.0, 20.0],
        default=10.0,
    )


def _locked_at_open(open_price, extreme_price, pct_chg, limit, upper):
    if upper:
        return (
            np.isclose(open_price, extreme_price, atol=1e-8)
            & (pct_chg >= pd.Series(limit, index=pct_chg.index) - 0.1)
        )
    return (
        np.isclose(open_price, extreme_price, atol=1e-8)
        & (pct_chg <= -pd.Series(limit, index=pct_chg.index) + 0.1)
    )


def evaluate_alpha(data, alpha_column, slippage_bp=10.0):
    valid = (
        data["base_eligible"]
        & data[alpha_column].notna()
        & np.isfinite(data[alpha_column])
    )
    sample = data.loc[valid].copy()
    sample = sample[
        sample.groupby("date")["code"].transform("size") >= MIN_NAMES
    ].copy()
    if sample.empty:
        return None, None

    ic_values = []
    for date, day in sample.groupby("date", sort=True):
        if len(day) < MIN_NAMES:
            continue
        if (
            day[alpha_column].nunique() < 2
            or day["exec_ret"].nunique() < 2
        ):
            continue
        ic_values.append({
            "date": date,
            "IC": day[alpha_column].corr(
                day["exec_ret"],
                method="spearman",
            ),
        })
    ic_series = pd.DataFrame(ic_values)
    ic_clean = ic_series["IC"].dropna()

    sample["layer"] = sample.groupby(
        "date",
        group_keys=False,
    )[alpha_column].transform(
        lambda s: pd.qcut(
            s.rank(method="first"),
            q=5,
            labels=["G1", "G2", "G3", "G4", "G5"],
        )
    )
    top = sample[sample["layer"] == "G5"].copy()
    top["weight"] = 1 / top.groupby("date")["code"].transform("size")

    weights = (
        top.pivot_table(
            index="date",
            columns="code",
            values="weight",
            aggfunc="sum",
        )
        .fillna(0.0)
        .sort_index()
    )
    returns = (
        top.pivot_table(
            index="date",
            columns="code",
            values="exec_ret",
            aggfunc="last",
        )
        .reindex(index=weights.index, columns=weights.columns)
    )
    if (weights.ne(0) & returns.isna()).any().any():
        raise ValueError(f"{alpha_column} 持仓存在缺失收益")

    returns = returns.fillna(0.0)
    gross_ret = (weights * returns).sum(axis=1)
    drifted = (
        weights.mul(1 + returns).div(1 + gross_ret, axis=0)
    )

    buys = np.empty(len(weights))
    sells = np.empty(len(weights))
    buys[0] = weights.iloc[0].clip(lower=0).sum()
    sells[0] = weights.iloc[0].clip(upper=0).abs().sum()
    delta = weights.iloc[1:].to_numpy() - drifted.iloc[:-1].to_numpy()
    buys[1:] = np.clip(delta, 0, None).sum(axis=1)
    sells[1:] = np.clip(-delta, 0, None).sum(axis=1)

    execution_dates = (
        sample.groupby("date")["entry_date"].first().reindex(weights.index)
    )
    stamp_bp = np.where(
        execution_dates < STAMP_CUT_DATE,
        10.0,
        5.0,
    )
    traded = buys + sells
    total_cost = (
        traded * (COMMISSION_BP + TRANSFER_BP + slippage_bp) / 1e4
        + sells * stamp_bp / 1e4
    )
    net_ret = gross_ret.to_numpy() - total_cost
    nav = (1 + net_ret).cumprod()
    drawdown = nav / np.maximum.accumulate(nav) - 1
    signal_dates = pd.Series(weights.index.to_numpy())
    period_stats = {
        "train": _period_metrics(
            signal_dates,
            net_ret,
            start=None,
            end=pd.Timestamp("2022-12-31"),
        ),
        "test": _period_metrics(
            signal_dates,
            net_ret,
            start=pd.Timestamp("2023-01-01"),
            end=None,
        ),
    }

    summary = {
        "factor": alpha_column,
        "days": len(net_ret),
        "avg_stocks": float(weights.ne(0).sum(axis=1).mean()),
        "IC_mean": float(ic_clean.mean()),
        "IC_std": float(ic_clean.std(ddof=1)),
        "IR": float(
            ic_clean.mean() / ic_clean.std(ddof=1)
        ),
        "positive_IC_rate": float((ic_clean > 0).mean()),
        "gross_annual_return": float(
            (1 + gross_ret).prod() ** (252 / len(gross_ret)) - 1
        ),
        "net_annual_return": float(
            nav[-1] ** (252 / len(net_ret)) - 1
        ),
        "net_annual_vol": float(
            np.std(net_ret, ddof=1) * np.sqrt(252)
        ),
        "net_sharpe": float(
            np.mean(net_ret)
            / np.std(net_ret, ddof=1)
            * np.sqrt(252)
        ),
        "max_drawdown": float(np.min(drawdown)),
        "avg_daily_turnover": float((traded / 2).mean()),
        "total_cost": float(total_cost.sum()),
        "slippage_bp": slippage_bp,
        "train_net_annual_return": period_stats["train"]["annual_return"],
        "train_net_sharpe": period_stats["train"]["sharpe"],
        "test_net_annual_return": period_stats["test"]["annual_return"],
        "test_net_sharpe": period_stats["test"]["sharpe"],
        "test_max_drawdown": period_stats["test"]["max_drawdown"],
    }
    curve = pd.DataFrame({
        "date": weights.index,
        "factor": alpha_column,
        "gross_ret": gross_ret.to_numpy(),
        "net_ret": net_ret,
        "nav": nav,
    })
    ic_series["factor"] = alpha_column
    return summary, (ic_series, curve)


def _period_metrics(dates, returns, start=None, end=None):
    mask = pd.Series(True, index=range(len(dates)))
    if start is not None:
        mask &= dates >= start
    if end is not None:
        mask &= dates <= end
    values = returns[mask.to_numpy()]
    if len(values) < 20 or np.std(values, ddof=1) == 0:
        return {
            "annual_return": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
        }
    nav = np.cumprod(1 + values)
    drawdown = nav / np.maximum.accumulate(nav) - 1
    return {
        "annual_return": float(
            nav[-1] ** (252 / len(values)) - 1
        ),
        "sharpe": float(
            np.mean(values)
            / np.std(values, ddof=1)
            * np.sqrt(252)
        ),
        "max_drawdown": float(np.min(drawdown)),
    }

