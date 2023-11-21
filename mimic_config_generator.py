import dataclasses
import itertools
import hashlib
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import yaml

from _config_feedstock import ConfigFeedstock


DEFAULT_SEED = 42


@dataclasses.dataclass
class MimicConfigFeedstock(ConfigFeedstock):
    CANCER_CCS_CODE: float
    COLS: Tuple
    MIN_SEQ_LEN: int = 16
    TFIDF_DOC_COL: str = None
    TFIDF_VOCAB_COL: str = None
    DIAGNOSIS_SELECTION: str = "full"
    DIAG_ONLY: bool = False
    PROC_ONLY: bool = False

    @property
    def config_name(self):
        return f"mimic_config_{self.hash[:7]}.yml"

    @property
    def experiment_name(self):
        return f"experiment_{self.hash[:7]}"

    def to_config(self):
        return self.to_kwargs()

    def generate_and_save_config(self, save_path: Path):
        config_name = save_path / self.config_name
        with open(config_name, "w") as yaml_file:
            yaml.dump(self.to_kwargs(), yaml_file)


def generate_catsyn_config(
    nb_patients: int,
    nb_states: int,
    nb_variables: int,
    nb_bins: int,
    distribution: str,
    distribution_kwargs: Dict,
    min_start_states: int,
    min_end_states: int,
    model: str,
    model_kwargs: Dict,
):
    return {
        "seed": DEFAULT_SEED,
        "patients": nb_patients,
        "variable_generator": {
            "nb_variables": nb_variables,
            "variable_kwargs": {
                "n_bins": nb_bins,
                "distribution": distribution,
                "distribution_kwargs": distribution_kwargs,
            },
        },
        "states_generator": {
            "nb_states": nb_states,
            "min_start_states": min_start_states,
            "min_end_states": min_end_states,
            "model": model,
            "model_kwargs": model_kwargs,
            "state_kwargs": {
                "persistence_dict": {
                    "min": 0.95,
                    "max": 0.95,
                }
            },
        },
    }


if __name__ == "__main__":
    feedstock = ConfigFeedstock(
        NB_PATIENTS=[100, 1000],
        NB_STATES=[50, 100],
    )
    for config in feedstock.permute_configs():
        print(config)
