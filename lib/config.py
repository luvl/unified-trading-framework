import typing
from typing import Tuple

import datetime

now = datetime.datetime.now()
date = str(now.day) + "." + str(now.month) + "_" + str(now.hour) + ":" + str(now.minute)

class DefaultMacro(typing.NamedTuple):
    WITH_EXTENDED_FEATURE: bool = True
    BUY: int = 1
    HOLD: int = 0
    SELL: int = -1
    REWARD_STD: float = 0.006
    TRAIN_W_VALIDATION: bool = False
    SEED: int = 17061996
    INIT_NO_OF_SHARES: int = 1000000
    AMOUNT: int = 0.05
    TRANSACTION_FEE: int = 0.002
    SELL_TRANSACTION_TAX: int = 0.001

class TrainConfig(typing.NamedTuple):
    epochs: int = 100 # 100
    steps: int = 500 # 500 # max steps until end
    window_length: int = 100
    mem_size: int = 10000
    nodes: int = 16
    folder: str = "log/" + str(epochs) + "_s:" + str(steps) + "_w:" + str(window_length) + "_" + date
    all_steps: int = epochs * steps
    percentage_explore: float = 0.8
    explore_steps: int = int(all_steps * percentage_explore)

class TestConfig(typing.NamedTuple):
    test_epochs: int = 1
    # test_steps: int = 500

class ValConfig(typing.NamedTuple):
    val_epochs: int = 2
    val_steps: int = 10

class EnvConfig(typing.NamedTuple):
    limit_data: int = 1500000
    action_one_hot: bool = True
    cost: float = 0.0001
    window: int = 10
    no_of_cluster: int = 2
    sentimental_feature: bool = True
    eps_feature: bool = True
    embedding_feature_len: int = 582 # 437 # 16092 pca
    env_name: str = "Deep Direct Reinforcement Learning"

class DataConfig(typing.NamedTuple):
    train_data: str
    test_data: str
    val_data: str

class HyperparamConfig(typing.NamedTuple):
    learning_rate: float = 0.001
    learning_rate_decay: float = 0.
    gamma: float = 0.95
    batch_size: int = 64
    tar_mod_up: float = 0.001
    delta_clip: int = 1

def init_config(parse_args) -> Tuple[EnvConfig, TrainConfig, TestConfig, ValConfig, HyperparamConfig, DefaultMacro]:
    env_config = EnvConfig()
    tra_config = TrainConfig()
    tes_config = TestConfig()
    val_config = ValConfig()
    par_config = HyperparamConfig()
    default_macro = DefaultMacro(parse_args.with_extended_feature)

    return env_config, tra_config, tes_config, val_config, par_config, default_macro