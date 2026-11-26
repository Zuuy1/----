"""Cross-sectional and time-series operators for Alpha101."""

import numpy as np
import pandas as pd


def cross_rank(df, column):
    return df.groupby("date", sort=False)[column].rank(
        method="average",
        pct=True,
    )


def delay(df, column, periods):
    return df.groupby("code", sort=False)[column].shift(periods)


def delta(df, column, periods):
    return (
        df[column]
        - delay(df, column, periods)
    )


def ts_sum(df, column, window):
    return df.groupby("code", sort=False)[column].transform(
        lambda s: s.rolling(window, min_periods=window).sum()
    )


def ts_min(df, column, window):
    return df.groupby("code", sort=False)[column].transform(
        lambda s: s.rolling(window, min_periods=window).min()
    )


def ts_max(df, column, window):
    return df.groupby("code", sort=False)[column].transform(
        lambda s: s.rolling(window, min_periods=window).max()
    )


def ts_std(df, column, window):
    return df.groupby("code", sort=False)[column].transform(
        lambda s: s.rolling(window, min_periods=window).std(ddof=0)
    )


def ts_rank(df, column, window):
    def last_rank(values):
        if np.isnan(values).any():
            return np.nan
        return (
            pd.Series(values)
            .rank(pct=True, method="average")
            .iloc[-1]
        )

    return df.groupby("code", sort=False)[column].transform(
        lambda s: s.rolling(window, min_periods=window).apply(
            last_rank,
            raw=True,
        )
    )


def ts_argmax(df, column, window):
    return df.groupby("code", sort=False)[column].transform(
        lambda s: s.rolling(window, min_periods=window).apply(
            lambda values: np.argmax(values) + 1,
            raw=True,
        )
    )


def ts_corr(df, left, right, window):
    parts = []
    for _, group in df.groupby("code", sort=False):
        parts.append(
            group[left].rolling(
                window,
                min_periods=window,
            ).corr(group[right])
        )
    return pd.concat(parts).sort_index()


def ts_cov(df, left, right, window):
    parts = []
    for _, group in df.groupby("code", sort=False):
        parts.append(
            group[left].rolling(
                window,
                min_periods=window,
            ).cov(group[right])
        )
    return pd.concat(parts).sort_index()


def signed_power(series, exponent):
    return np.sign(series) * np.abs(series) ** exponent

