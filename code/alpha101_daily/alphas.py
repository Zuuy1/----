"""Twenty Alpha101 formulas that require only daily OHLCV data."""

import numpy as np
import pandas as pd

from .operators import (
    cross_rank,
    delay,
    delta,
    signed_power,
    ts_argmax,
    ts_corr,
    ts_cov,
    ts_max,
    ts_min,
    ts_rank,
    ts_std,
    ts_sum,
)


def prepare_features(df):
    data = df.sort_values(["code", "date"]).reset_index(drop=True).copy()
    data["ret"] = data["close"] / data["preclose"] - 1
    data["vwap"] = (
        data["amount"] / data["volume"].where(data["volume"] > 0)
    )
    data["adv20"] = ts_sum(data, "volume", 20) / 20
    return data


def build_alpha101(df):
    data = prepare_features(df)
    alphas = {}

    downside_std = ts_std(data, "ret", 20)
    alpha001_input = pd.Series(
        np.where(
            data["ret"] < 0,
            downside_std,
            data["close"],
        ),
        index=data.index,
    )
    alphas["alpha001"] = (
        cross_rank(
            data.assign(
                _a=ts_argmax(
                    data.assign(
                        _x=signed_power(alpha001_input, 2)
                    ),
                    "_x",
                    5,
                )
            ),
            "_a",
        )
        - 0.5
    )

    rank_log_volume = cross_rank(
        data.assign(_x=np.log(data["volume"].where(data["volume"] > 0))),
        "_x",
    )
    rank_log_volume_delta = cross_rank(
        data.assign(_x=delta(
            data.assign(
                _logv=np.log(data["volume"].where(data["volume"] > 0))
            ),
            "_logv",
            2,
        )),
        "_x",
    )
    rank_open_close = cross_rank(
        data.assign(_x=(data["close"] - data["open"]) / data["open"]),
        "_x",
    )
    alphas["alpha002"] = -ts_corr(
        data.assign(
            _left=rank_log_volume_delta,
            _right=rank_open_close,
        ),
        "_left",
        "_right",
        6,
    )

    ranked_open = cross_rank(data.assign(_x=data["open"]), "_x")
    ranked_volume = cross_rank(data.assign(_x=data["volume"]), "_x")
    alphas["alpha003"] = -ts_corr(
        data.assign(_left=ranked_open, _right=ranked_volume),
        "_left",
        "_right",
        10,
    )
    ranked_low = cross_rank(data.assign(_x=data["low"]), "_x")
    alphas["alpha004"] = -ts_rank(
        data.assign(_x=ranked_low),
        "_x",
        9,
    )

    vwap_advance = ts_sum(data, "vwap", 10) / 10
    alphas["alpha005"] = (
        cross_rank(
            data.assign(_x=data["open"] - vwap_advance),
            "_x",
        )
        * -cross_rank(
            data.assign(_x=data["close"] - data["vwap"]),
            "_x",
        ).abs()
    )
    alphas["alpha006"] = -ts_corr(data, "open", "volume", 10)

    close_change_7 = delta(data, "close", 7)
    alpha007_direction = (
        -ts_rank(
            data.assign(_x=close_change_7.abs()),
            "_x",
            60,
        )
        * np.sign(close_change_7)
    )
    alphas["alpha007"] = pd.Series(
        np.where(
            data["adv20"] < data["volume"],
            alpha007_direction,
            -1.0,
        ),
        index=data.index,
    )

    open_return_product = (
        ts_sum(data, "open", 5) * ts_sum(data, "ret", 5)
    )
    alphas["alpha008"] = -cross_rank(
        data.assign(_x=open_return_product),
        "_x",
    ) + cross_rank(
        data.assign(
            _x=delay(
                data.assign(_p=open_return_product),
                "_p",
                10,
            )
        ),
        "_x",
    )

    close_delta_1 = delta(data, "close", 1)
    data["_delta_close"] = close_delta_1
    min_change_5 = data.groupby("code", sort=False)[
        "_delta_close"
    ].transform(lambda s: s.rolling(5, min_periods=5).min())
    max_change_5 = data.groupby("code", sort=False)[
        "_delta_close"
    ].transform(lambda s: s.rolling(5, min_periods=5).max())
    alphas["alpha009"] = pd.Series(
        np.where(
            min_change_5 > 0,
            close_delta_1,
            np.where(
                max_change_5 < 0,
                close_delta_1,
                -close_delta_1,
            ),
        ),
        index=data.index,
    )

    min_change_4 = data.groupby("code", sort=False)[
        "_delta_close"
    ].transform(lambda s: s.rolling(4, min_periods=4).min())
    max_change_4 = data.groupby("code", sort=False)[
        "_delta_close"
    ].transform(lambda s: s.rolling(4, min_periods=4).max())
    alpha010_input = pd.Series(
        np.where(
            min_change_4 > 0,
            close_delta_1,
            np.where(
                max_change_4 < 0,
                close_delta_1,
                -close_delta_1,
            ),
        ),
        index=data.index,
    )
    alphas["alpha010"] = cross_rank(
        data.assign(_x=alpha010_input),
        "_x",
    )

    vwap_close = data["vwap"] - data["close"]
    volume_delta_3 = delta(data, "volume", 3)
    alphas["alpha011"] = (
        cross_rank(
            data.assign(_x=ts_max(
                data.assign(_v=vwap_close),
                "_v",
                3,
            )),
            "_x",
        )
        + cross_rank(
            data.assign(_x=ts_min(
                data.assign(_v=vwap_close),
                "_v",
                3,
            )),
            "_x",
        )
    ) * cross_rank(
        data.assign(_x=volume_delta_3),
        "_x",
    )
    alphas["alpha012"] = (
        np.sign(delta(data, "volume", 1)) * -close_delta_1
    )

    ranked_close = cross_rank(data.assign(_x=data["close"]), "_x")
    ranked_high = cross_rank(data.assign(_x=data["high"]), "_x")
    alphas["alpha013"] = -cross_rank(
        data.assign(_x=ts_cov(
            data.assign(_l=ranked_close, _r=ranked_volume),
            "_l",
            "_r",
            5,
        )),
        "_x",
    )
    alphas["alpha014"] = (
        -cross_rank(
            data.assign(_x=delta(data.assign(_r=data["ret"]), "_r", 3)),
            "_x",
        )
        * ts_corr(data, "open", "volume", 10)
    )
    alphas["alpha015"] = -ts_sum(
        data.assign(_x=cross_rank(
            data.assign(_c=ts_corr(
                data.assign(_l=ranked_high, _r=ranked_volume),
                "_l",
                "_r",
                3,
            )),
            "_c",
        )),
        "_x",
        3,
    )
    alphas["alpha016"] = -cross_rank(
        data.assign(_x=ts_cov(
            data.assign(_l=ranked_high, _r=ranked_volume),
            "_l",
            "_r",
            5,
        )),
        "_x",
    )

    alpha018_input = (
        ts_std(
            data.assign(_x=(data["close"] - data["open"]).abs()),
            "_x",
            5,
        )
        + (data["close"] - data["open"])
        + ts_corr(data, "close", "open", 10)
    )
    alphas["alpha018"] = -cross_rank(
        data.assign(_x=alpha018_input),
        "_x",
    )
    alphas["alpha020"] = (
        -cross_rank(
            data.assign(_x=data["open"] - delay(data, "high", 1)),
            "_x",
        )
        * cross_rank(
            data.assign(_x=data["open"] - delay(data, "close", 1)),
            "_x",
        )
        * cross_rank(
            data.assign(_x=data["open"] - delay(data, "low", 1)),
            "_x",
        )
    )

    alphas["alpha022"] = (
        -delta(
            data.assign(_x=ts_corr(data, "high", "volume", 5)),
            "_x",
            5,
        )
        * cross_rank(
            data.assign(_x=ts_std(data, "close", 20)),
            "_x",
        )
    )
    alphas["alpha025"] = cross_rank(
        data.assign(
            _x=(
                (-data["ret"] * data["adv20"] * data["vwap"])
                * (data["high"] - data["close"])
            )
        ),
        "_x",
    )

    output = pd.DataFrame(alphas)
    output.insert(0, "code", data["code"])
    output.insert(0, "date", data["date"])
    return data, output


ALPHA_COLUMNS = [
    f"alpha{number:03d}"
    for number in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                   11, 12, 13, 14, 15, 16, 18, 20, 22, 25]
]
