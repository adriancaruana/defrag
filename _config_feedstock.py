import dataclasses
import itertools
import hashlib
from pathlib import Path
from typing import Dict, Tuple, Optional

import yaml


class ConfigFeedstock:
    def permute_configs(self):
        iterable_attrs = []
        attrs_dict = self.to_attrs()
        for k, v in attrs_dict.items():
            if isinstance(v, list):
                iterable_attrs.append(k)

        iterable_values = itertools.product(
            *list(attrs_dict[k] for k in iterable_attrs)
        )
        for permutation in iterable_values:
            permutation_kwargs = {k: v for k, v in zip(iterable_attrs, permutation)}
            _derived = self.__class__  # ConfigFeedstock instance may be a derived class
            yield _derived(**{**self.to_attrs(), **permutation_kwargs})

    @property
    def hash(self):
        return hashlib.md5(str(self.to_attrs()).encode("utf-8")).hexdigest()

    @property
    def config_name(self):
        return f"config_permutation_{self.hash[:7]}.yml"

    @property
    def experiment_name(self):
        return f"experiment_{self.hash[:7]}"

    def to_attrs(self):
        return {
            k: v
            for k, v in self.__dict__.items()
            if k[:1] != "_" and not isinstance(v, classmethod)
        }

    def to_kwargs(self):
        return {k.lower(): v for k, v in self.to_attrs().items()}

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
                    "min": 0.9,
                    "max": 0.9,
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
