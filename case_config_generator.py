import dataclasses
import itertools
import hashlib
from pathlib import Path
from typing import Dict, Tuple, Optional

import yaml

from case.case import Cat2VecConfig, Seq2SeqConfig
from _config_feedstock import ConfigFeedstock


@dataclasses.dataclass
class Cat2VecConfigFeedstock(ConfigFeedstock):
    MODEL_TYPE: str = "AEModel"
    N_COL_FEATURES: int = 64
    N_STATE_FEATURES: int = 64
    COLS: Tuple = ("VAR_0", "VAR_1", "VAR_2")
    TRAIN_STEPS: int = 5_000
    # TRAIN_STEPS: int = 200

    @property
    def config_name(self):
        return f"cat2vec_config_{self.hash[:7]}.yml"

    @property
    def experiment_name(self):
        return f"experiment_{self.hash[:7]}"

    def to_config(self):
        return Cat2VecConfig(**self.to_kwargs())

    def generate_and_save_config(self, save_path: Path):
        config_name = save_path / self.config_name
        with open(config_name, "w") as yaml_file:
            yaml.dump(self.to_kwargs(), yaml_file)


@dataclasses.dataclass
class Seq2SeqConfigFeedstock(ConfigFeedstock):
    MODEL_TYPE: str = "Seq2Seq"
    COLS: Tuple = ("VAR_0", "VAR_1", "VAR_2")
    # This is the sequence length used during training
    L: int = 8
    # d_model: This must be the same as Cat2VecConfigFeedstock.N_STATE_FEATURES
    M: int = 128
    # num_heads: M must be divisible by N (d_model % num_heads)
    N: int = 16
    NUM_ENCODER_LAYERS: int = 8
    NUM_DECODER_LAYERS: int = 8
    DIM_FEEDFORWARD: int = 64
    DROPOUT: float = 0.2
    BATCH_SIZE: int = 64
    TRAIN_STEPS: int = 50_000
    VALIDATION_STEPS: int = 25
    P_MASK_THRESH: float = 0.0
    ENCODER_WINDOW_SIZE: int = 5
    # TRAIN_STEPS: int = 200
    # VALIDATION_STEPS: int = 2

    @property
    def config_name(self):
        return f"seq2seq_config_{self.hash[:7]}.yml"

    @property
    def experiment_name(self):
        return f"experiment_{self.hash[:7]}"

    def to_config(self):
        return Seq2SeqConfig(**self.to_kwargs())

    def generate_and_save_config(self, save_path: Path):
        config_name = save_path / self.config_name
        with open(config_name, "w") as yaml_file:
            yaml.dump(self.to_kwargs(), yaml_file)
