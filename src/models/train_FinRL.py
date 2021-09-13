import sys

sys.path.append("../FinRL")


import argparse
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# matplotlib.use('Agg')
import datetime

from finrl.apps import config
from finrl.neo_finrl.preprocessor.yahoodownloader import YahooDownloader
from finrl.neo_finrl.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.neo_finrl.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.neo_finrl.env_stock_trading.env_stocktrading_np import (
    StockTradingEnv as StockTradingEnv_numpy,
)
from finrl.drl_agents.stablebaselines3.models import DRLAgent
from finrl.drl_agents.rllib.models import DRLAgent as DRLAgent_rllib

from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
import ray
from pprint import pprint
import os
import itertools
import glob
from os.path import join, exists
from sklearn.model_selection import train_test_split
import datetime as dt

from src.data import load_OHLCV_files
from src.data.preparation import TRAIN_SIZE


def main():
    out_dir = join("output", "trading", "FinRL")
    if not exists(out_dir):
        os.makedirs(out_dir)

    cli = argparse.ArgumentParser()
    cli.add_argument("year", type=str)
    cli.add_argument("model", type=str)
    cli.add_argument("--timesteps", type=int, default=50000)
    cli.add_argument("--overwrite", action="store_true")
    args = cli.parse_args()

    year = args.year
    model = args.model
    timesteps = args.timesteps

    sim_id = f"{model}_{year}"

    if exists(join(out_dir, f"{sim_id}_stats.csv")) and not args.overwrite:
        print(f"Results already exists, skipping {sim_id}")
        return

    if not os.path.exists("./" + config.DATA_SAVE_DIR):
        os.makedirs("./" + config.DATA_SAVE_DIR)
    if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
        os.makedirs("./" + config.TRAINED_MODEL_DIR)
    if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
        os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
    if not os.path.exists("./" + config.RESULTS_DIR):
        os.makedirs("./" + config.RESULTS_DIR)

    ticks = [f.split(".")[0] for f in os.listdir("data/processed/SP500_technical")]
    print("Ticks", ticks[:5], len(ticks))

    stock_by_tick = load_OHLCV_files("data/processed/SP500_technical")
    print(len(stock_by_tick))

    # reuse the FinRL format
    stock_dfs = list()
    for tick, df in stock_by_tick.items():
        df = df.loc[year]
        df["tic"] = tick
        stock_dfs.append(df)

    df = pd.concat(stock_dfs)

    print(df.shape)

    fe = FeatureEngineer(
        use_technical_indicator=False,
        use_turbulence=False,
        user_defined_feature=False,
        use_vix=True,
    )

    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]

    df["date"] = df["date"].astype(str)
    print("Start processing", flush=True)
    processed = fe.preprocess_data(df)
    print("Processed", processed.shape, flush=True)

    list_ticker = processed["tic"].unique().tolist()
    list_date = list(
        pd.date_range(processed["date"].min(), processed["date"].max()).astype(str)
    )
    combination = list(itertools.product(list_date, list_ticker))

    processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(
        processed, on=["date", "tic"], how="left"
    )
    processed_full = processed_full[processed_full["date"].isin(processed["date"])]
    processed_full = processed_full.sort_values(["date", "tic"])

    processed_full = processed_full.fillna(0)
    print("Processed full", processed_full.shape, flush=True)

    #  align training / trading days with other simulations
    train_df, trade_df = train_test_split(
        stock_by_tick["AAPL"].loc[year], train_size=TRAIN_SIZE, shuffle=False
    )

    start_train = str(train_df.index[0].date())
    end_train = str(trade_df.index[0].date())
    start_trade = end_train
    end_trade = str(trade_df.index[-1].date() + dt.timedelta(days=1))

    print(start_train, end_train, flush=True)
    print(start_trade, end_trade, flush=True)

    print("Dates")
    train = data_split(processed, start_train, end_train)
    trade = data_split(processed, start_trade, end_trade)

    indicators_list = [
        "sma5-20",
        "sma8-15",
        "sma20-50",
        "ema5-20",
        "ema8-15",
        "ema20-50",
        "macd12-26",
        "ao14",
        "adx14",
        "wd14",
        "ppo12_26",
        "rsi14",
        "mfi14",
        "tsi",
        "so14",
        "cmo14",
        "atrp14",
        "pvo14",
        "fi13",
        "fi50",
        "adl",
        "obv",
    ]

    stock_dimension = len(train.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(indicators_list) * stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    env_kwargs = {
        "hmax": 100,
        "initial_amount": 100000,
        "buy_cost_pct": 0.001,
        "sell_cost_pct": 0.001,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": indicators_list,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
    }
    print(env_kwargs)

    e_train_gym = StockTradingEnv(df=train, **env_kwargs)

    env_train, _ = e_train_gym.get_sb_env()
    agent = DRLAgent(env=env_train)

    print("Getting model...")
    if model == "A2C":
        model_ = agent.get_model("a2c")
    elif model == "DDPG":
        model_ = agent.get_model("ddpg")
    elif model == "PPO":
        PPO_PARAMS = {
            "n_steps": 2048,
            "ent_coef": 0.01,
            "learning_rate": 0.00025,
            "batch_size": 128,
        }
        model_ = agent.get_model("ppo", model_kwargs=PPO_PARAMS)
    elif model == "TD3":
        TD3_PARAMS = {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001}
        model_ = agent.get_model("td3", model_kwargs=TD3_PARAMS)
    elif model == "SAC":
        SAC_PARAMS = {
            "batch_size": 128,
            "buffer_size": 1000000,
            "learning_rate": 0.0001,
            "learning_starts": 100,
            "ent_coef": "auto_0.1",
        }
        model_ = agent.get_model("sac", model_kwargs=SAC_PARAMS)
    else:
        raise ValueError(f"Unsupported model: {model}")

    print("Training...")
    trained_model = agent.train_model(
        model=model_, tb_log_name=sim_id, total_timesteps=timesteps
    )
    print("Training completed!")

    print("Doing trades 💸")
    # breakpoint()

    e_trade_gym = StockTradingEnv(df=trade, risk_indicator_col="vix", **env_kwargs)

    df_account_value, df_actions = DRLAgent.DRL_prediction(
        model=trained_model, environment=e_trade_gym
    )

    df_account_value.to_csv(join(out_dir, f"{sim_id}_account_value.csv"))
    df_actions.to_csv(join(out_dir, f"{sim_id}_actions.csv"))

    print("Backtesting")
    perf = backtest_stats(account_value=df_account_value)
    perf = pd.DataFrame(perf)
    perf.to_csv(join(out_dir, f"{sim_id}_stats.csv"))


if __name__ == "__main__":
    main()
