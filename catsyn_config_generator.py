import dataclasses
import itertools
import hashlib
from pathlib import Path
from typing import Dict, Tuple, Optional

import yaml

from _config_feedstock import ConfigFeedstock


DEFAULT_SEED = 42
DEFAULT_DISTRIBUTION = "zipf_shuffle"
DEFAULT_MODEL = "directed_extended_barabasi_albert_graph"

distribution_default_kwargs = {
    "zipf_shuffle": {"a": 2},
}
model_default_kwargs = {
    "directed_extended_barabasi_albert_graph": {
        "n": 50,
        "m": 1,
        "p": 0.1,
        "q": 0,
    },
    "gnp_random_graph": {
        "n": 50,
        "p": 0.2,
    },
}


@dataclasses.dataclass
class CatsynConfigFeedstock(ConfigFeedstock):
    NB_PATIENTS: int = 1000
    NB_STATES: int = 50
    NB_VARIABLES: int = 1
    NB_BINS: int = 100
    DISTRIBUTION: Tuple[str, Optional[Dict]] = (
        "zipf_shuffle",
        distribution_default_kwargs["zipf_shuffle"],
    )
    DISTRIBUTION_KWARGS: Optional[Dict] = None
    MIN_START_STATES: Optional[int] = 1
    MIN_END_STATES: Optional[int] = 1
    MODEL: Tuple[str, Optional[Dict]] = (
        "gnp_random_graph",
        model_default_kwargs["gnp_random_graph"],
    )
    MODEL_KWARGS: Optional[Dict] = None
    PERSISTENCE_MAX: Optional[float] = 0.95
    PERSISTENCE_MIN: Optional[float] = 0.95
    SEED: Optional[int] = DEFAULT_SEED

    def __post_init__(self):
        if isinstance(self.DISTRIBUTION, tuple):
            self.DISTRIBUTION_KWARGS = self.DISTRIBUTION[1]
            self.DISTRIBUTION = self.DISTRIBUTION[0]
            if self.DISTRIBUTION_KWARGS is None:
                self.DISTRIBUTION_KWARGS = distribution_default_kwargs[
                    self.DISTRIBUTION
                ]
        if isinstance(self.MODEL, tuple):
            self.MODEL_KWARGS = self.MODEL[1]
            self.MODEL = self.MODEL[0]
            if self.MODEL_KWARGS is None:
                self.MODEL_KWARGS = {**model_default_kwargs[self.MODEL], **{"seed": DEFAULT_SEED}}

    @property
    def config_name(self):
        return f"catsyn_config_{self.hash[:7]}.yml"

    @property
    def experiment_name(self):
        return f"experiment_{self.hash[:7]}"

    def to_config(self):
        return generate_catsyn_config(**self.to_kwargs())

    def generate_and_save_config(self, save_path: Path):
        config_name = save_path / self.config_name
        with open(config_name, "w") as yaml_file:
            yaml.dump(self.to_config(), yaml_file)


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
    persistence_min: float,
    persistence_max: float,
    seed: int,
):
    return {
        "seed": seed,
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
                    "min": persistence_min,
                    "max": persistence_max,
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
