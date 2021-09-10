import pandas as pd
import numpy as np


UP = 2
DOWN = 0

DEF = 1


TA_DISAGREEMENT = {
    "SMA5-20": {UP: {"(5:inf)": 1}, DOWN: {"(-inf,-5]": 1}},
    "SMA8-15": {UP: {"(5:inf)": 1}, DOWN: {"(-inf,-5]": 1}},
    "SMA20-50": {UP: {"(5:inf)": 1}, DOWN: {"(-inf,-5]": 1}},
    "EMA5-20": {UP: {"(5:inf)": 1}, DOWN: {"(-inf,-5]": 1}},
    "EMA8-15": {UP: {"(5:inf)": 1}, DOWN: {"(-inf,-5]": 1}},
    "EMA20-50": {UP: {"(5:inf)": 1}, DOWN: {"(-inf,-5]": 1}},
    "MACD12-26": {UP: {"(5:inf)": 1}, DOWN: {"(-inf,-5]": 1}},
    "AO14": {UP: {"[-100:-50]": 1, "(-50:0]": 1}, DOWN: {"(0:50]": 1, "(50:100]": 1}},
    "ADX14": {UP: {}, DOWN: {}},
    "WD14": {UP: {}, DOWN: {}},
    "PPO12_26": {UP: {"(-inf:-5]": 1, "(-5:0]": 1}, DOWN: {"(0:5]": 1, "(5:inf)": 1}},
    "RSI14": {UP: {"(70,85]": 1, "(85,100]": 2}, DOWN: {"(0,15]": 2, "(15,30]": 1}},
    "MFI14": {UP: {}, DOWN: {}},
    "TSI": {UP: {"(25:inf)": 1}, DOWN: {"(-inf:-25]"}},
    "SO14": {UP: {}, DOWN: {}},
    "CMO14": {
        UP: {"(50:75]": 1, "(75:100]": 1},
        DOWN: {"[-100:-75]": 1, "(-75:-50]": 1},
    },
    "ATRP14": {UP: {}, DOWN: {}},
    "PVO14": {UP: {}, DOWN: {}},
    "ADL": {UP: {}, DOWN: {}},
    "OBV": {UP: {}, DOWN: {}},
    "FI13": {UP: {}, DOWN: {}},
    "FI50": {UP: {}, DOWN: {}},
}


def discretize_technical_columns(df):
    """Perform two types of discretization strategies."""
    classic_disc = list()
    express_disc = list()

    sep = ":"

    classic_disc.append(
        pd.cut(
            df["SMA5-20"],
            [-np.inf, 0, np.inf],
            labels=[f"(-inf{sep}0]", f"(0{sep}inf)"],
        )
    )
    express_disc.append(
        pd.cut(
            df["SMA5-20"],
            [-np.inf, -5, 0, 5, np.inf],
            labels=[f"(-inf{sep}-5]", f"(-5{sep}0]", f"(0{sep}5]", f"(5{sep}inf)"],
        )
    )

    classic_disc.append(
        pd.cut(
            df["SMA8-15"],
            [-np.inf, 0, np.inf],
            labels=[f"(-inf{sep}0]", f"(0{sep}inf)"],
        )
    )
    express_disc.append(
        pd.cut(
            df["SMA8-15"],
            [-np.inf, -5, 0, 5, np.inf],
            labels=[f"(-inf{sep}-5]", f"(-5{sep}0]", f"(0{sep}5]", f"(5{sep}inf)"],
        )
    )

    classic_disc.append(
        pd.cut(
            df["SMA20-50"],
            [-np.inf, 0, np.inf],
            labels=[f"(-inf{sep}0]", f"(0{sep}inf)"],
        )
    )
    express_disc.append(
        pd.cut(
            df["SMA20-50"],
            [-np.inf, -5, 0, 5, np.inf],
            labels=[f"(-inf{sep}-5]", f"(-5{sep}0]", f"(0{sep}5]", f"(5{sep}inf)"],
        )
    )

    classic_disc.append(
        pd.cut(
            df["EMA5-20"],
            [-np.inf, 0, np.inf],
            labels=[f"(-inf{sep}0]", f"(0{sep}inf)"],
        )
    )
    express_disc.append(
        pd.cut(
            df["EMA5-20"],
            [-np.inf, -5, 0, 5, np.inf],
            labels=[f"(-inf{sep}-5]", f"(-5{sep}0]", f"(0{sep}5]", f"(5{sep}inf)"],
        )
    )

    classic_disc.append(
        pd.cut(
            df["EMA8-15"],
            [-np.inf, 0, np.inf],
            labels=[f"(-inf{sep}0]", f"(0{sep}inf)"],
        )
    )
    express_disc.append(
        pd.cut(
            df["EMA8-15"],
            [-np.inf, -5, 0, 5, np.inf],
            labels=[f"(-inf{sep}-5]", f"(-5{sep}0]", f"(0{sep}5]", f"(5{sep}inf)"],
        )
    )

    classic_disc.append(
        pd.cut(
            df["EMA20-50"],
            [-np.inf, 0, np.inf],
            labels=[f"(-inf{sep}0]", f"(0{sep}inf)"],
        )
    )
    express_disc.append(
        pd.cut(
            df["EMA20-50"],
            [-np.inf, -5, 0, 5, np.inf],
            labels=[f"(-inf{sep}-5]", f"(-5{sep}0]", f"(0{sep}5]", f"(5{sep}inf)"],
        )
    )

    classic_disc.append(
        pd.cut(
            df["MACD12-26"],
            [-np.inf, 0, np.inf],
            labels=[f"(-inf{sep}0]", f"(0{sep}inf)"],
        )
    )
    express_disc.append(
        pd.cut(
            df["MACD12-26"],
            [-np.inf, -5, -2, 0, 2, 5, np.inf],
            labels=[
                f"(-inf{sep}-5]",
                f"(-5{sep}-2]",
                f"(-2{sep}0]",
                f"(0{sep}2]",
                f"(2{sep}5]",
                f"(5{sep}inf)",
            ],
        )
    )

    classic_disc.append(
        pd.cut(
            df["AO14"],
            [-100, 0, 100],
            labels=[f"[-100{sep}0]", f"(0{sep}100]"],
            include_lowest=True,
        )
    )
    express_disc.append(
        pd.cut(
            df["AO14"],
            [-100, -50, 0, 50, 100],
            labels=[f"[-100{sep}-50]", f"(-50{sep}0]", f"(0{sep}50]", f"(50{sep}100]"],
            include_lowest=True,
        )
    )

    classic_disc.append(
        pd.cut(
            df["ADX14"],
            [-np.inf, 20, 25, np.inf],
            labels=[f"(-inf{sep}20]", f"(20{sep}25]", f"(25{sep}inf)"],
        )
    )
    express_disc.append(
        pd.cut(
            df["ADX14"],
            [-np.inf, 20, 25, 40, np.inf],
            labels=[f"(-inf{sep}20]", f"(20{sep}25]", f"(25{sep}40]", f"(40{sep}inf)"],
        )
    )

    classic_disc.append(
        pd.cut(
            df["WD14"], [-np.inf, 0, np.inf], labels=[f"(-inf{sep}0]", f"(0{sep}inf)"]
        )
    )
    express_disc.append(
        pd.cut(
            df["WD14"],
            [-np.inf, -5, 0, 5, np.inf],
            labels=[f"(-inf{sep}-5]", f"(-5{sep}0]", f"(0{sep}5]", f"(5{sep}inf)"],
        )
    )

    classic_disc.append(
        pd.cut(
            df["PPO12_26"],
            [-np.inf, 0, np.inf],
            labels=[f"(-inf{sep}0]", f"(0{sep}inf)"],
        )
    )
    express_disc.append(
        pd.cut(
            df["PPO12_26"],
            [-np.inf, -5, 0, 5, np.inf],
            labels=[f"(-inf{sep}-5]", f"(-5{sep}0]", f"(0{sep}5]", f"(5{sep}inf)"],
        )
    )

    classic_disc.append(
        pd.cut(
            df["RSI14"],
            [0, 30, 70, 100],
            labels=[f"[0{sep}30]", f"(30{sep}70]", f"(70{sep}100]"],
            include_lowest=True,
        )
    )
    express_disc.append(
        pd.cut(
            df["RSI14"],
            [0, 15, 30, 50, 70, 85, 100],
            labels=[
                f"[0{sep}15]",
                f"(15{sep}30]",
                f"(30{sep}50]",
                f"(50{sep}70]",
                f"(70{sep}85]",
                f"(85{sep}100]",
            ],
            include_lowest=True,
        )
    )

    classic_disc.append(
        pd.cut(
            df["MFI14"],
            [0, 30, 70, 100],
            labels=[f"[0{sep}30]", f"(30{sep}70]", f"(70{sep}100]"],
            include_lowest=True,
        )
    )
    express_disc.append(
        pd.cut(
            df["MFI14"],
            [0, 15, 30, 50, 70, 85, 100],
            labels=[
                f"[0{sep}15]",
                f"(15{sep}30]",
                f"(30{sep}50]",
                f"(50{sep}70]",
                f"(70{sep}85]",
                f"(85{sep}100]",
            ],
            include_lowest=True,
        )
    )

    classic_disc.append(
        pd.cut(
            df["TSI"],
            [-np.inf, -25, 25, np.inf],
            labels=[f"(-inf{sep}-25]", f"(-25{sep}25]", f"(25{sep}inf)"],
        )
    )
    express_disc.append(
        pd.cut(
            df["TSI"],
            [-np.inf, -25, 0, 25, np.inf],
            labels=[f"(-inf{sep}-25]", f"(-25{sep}0]", f"(0{sep}25]", f"(25{sep}inf)"],
        )
    )

    classic_disc.append(
        pd.cut(
            df["SO14"],
            [0, 20, 80, 100],
            labels=[f"[0{sep}20]", f"(20{sep}80]", f"(80{sep}100]"],
            include_lowest=True,
        )
    )
    express_disc.append(
        pd.cut(
            df["SO14"],
            [0, 10, 20, 50, 80, 90, 100],
            labels=[
                f"[0{sep}10]",
                f"(10{sep}20]",
                f"(20{sep}50]",
                f"(50{sep}80]",
                f"(80{sep}90]",
                f"(90{sep}100]",
            ],
            include_lowest=True,
        )
    )

    classic_disc.append(
        pd.cut(
            df["CMO14"],
            [-100, -50, 50, 100],
            labels=[f"[-100{sep}-50]", f"(-50{sep}50]", f"(50{sep}100]"],
            include_lowest=True,
        )
    )
    express_disc.append(
        pd.cut(
            df["CMO14"],
            [-100, -75, -50, 0, 50, 75, 100],
            labels=[
                f"[-100{sep}-75]",
                f"(-75{sep}-50]",
                f"(-50{sep}0]",
                f"(0{sep}50]",
                f"(50{sep}75]",
                f"(75{sep}100]",
            ],
            include_lowest=True,
        )
    )

    classic_disc.append(
        pd.cut(
            df["ATRP14"],
            [0, 30, 100],
            labels=[f"[0{sep}30]", f"(30{sep}100]"],
            include_lowest=True,
        )
    )
    express_disc.append(
        pd.cut(
            df["ATRP14"],
            [0, 10, 30, 40, 100],
            labels=[f"[0{sep}10]", f"(10{sep}30]", f"(30{sep}40]", f"(40{sep}100]"],
            include_lowest=True,
        )
    )

    classic_disc.append(
        pd.cut(
            df["PVO14"],
            [-100, 0, 100],
            labels=[f"[-100{sep}0]", f"(0{sep}100]"],
            include_lowest=True,
        )
    )
    express_disc.append(
        pd.cut(
            df["PVO14"],
            [-100, -40, -20, 0, 20, 40, 100],
            labels=[
                f"[-100{sep}-40]",
                f"(-40{sep}-20]",
                f"(-20{sep}0]",
                f"(0{sep}20]",
                f"(20{sep}40]",
                f"(40{sep}100]",
            ],
            include_lowest=True,
        )
    )

    classic_disc.append(
        pd.cut(
            df["ADL"], [-np.inf, 0, np.inf], labels=[f"(-inf{sep}0]", f"(0{sep}inf)"]
        )
    )
    express_disc.append(
        pd.cut(
            df["ADL"],
            [-np.inf, -1e9, 0, 1e9, np.inf],
            labels=[
                f"(-inf{sep}-1e9]",
                f"(-1e9{sep}0]",
                f"(0{sep}1e9]",
                f"(1e9{sep}inf)",
            ],
        )
    )

    classic_disc.append(
        pd.cut(
            df["OBV"], [-np.inf, 0, np.inf], labels=[f"(-inf{sep}0]", f"(0{sep}inf)"]
        )
    )
    express_disc.append(
        pd.cut(
            df["OBV"],
            [-np.inf, -1e9, 0, 1e9, np.inf],
            labels=[
                f"(-inf{sep}-1e9]",
                f"(-1e9{sep}0]",
                f"(0{sep}1e9]",
                f"(1e9{sep}inf)",
            ],
        )
    )

    classic_disc.append(
        pd.cut(
            df["FI13"], [-np.inf, 0, np.inf], labels=[f"(-inf{sep}0]", f"(0{sep}inf)"]
        )
    )
    express_disc.append(
        pd.cut(
            df["FI13"],
            [-np.inf, -1e7, 0, 1e7, np.inf],
            labels=[
                f"(-inf{sep}-1e7]",
                f"(-1e7{sep}0]",
                f"(0{sep}1e7]",
                f"(1e7{sep}inf)",
            ],
        )
    )

    classic_disc.append(
        pd.cut(
            df["FI50"], [-np.inf, 0, np.inf], labels=[f"(-inf{sep}0]", f"(0{sep}inf)"]
        )
    )
    express_disc.append(
        pd.cut(
            df["FI50"],
            [-np.inf, -1e7, 0, 1e7, np.inf],
            labels=[
                f"(-inf{sep}-1e7]",
                f"(-1e7{sep}0]",
                f"(0{sep}1e7]",
                f"(1e7{sep}inf)",
            ],
        )
    )

    return pd.concat(classic_disc, axis=1), pd.concat(express_disc, axis=1)
