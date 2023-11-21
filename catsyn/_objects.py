import dataclasses
import logging
from pathlib import Path
from tkinter import Y
from typing import Callable, Dict, List, Tuple, Union
import yaml

import numpy as np
import pandas as pd
from matplotlib import cm
import networkx as nx
from graph_tool.draw import graph_draw as gt_graph_draw
import scipy.stats
from tqdm import tqdm

from ._misc import str_to_state, PhonemicName
from ._model_factory import ModelFactory, nx2gt

GLOBAL_RANDOM_STATE = None

import matplotlib
matplotlib.use("Agg")

MAX_NUM_EVENTS = 1000

def set_global_random_state(seed: int = 42):
    global GLOBAL_RANDOM_STATE
    GLOBAL_RANDOM_STATE = np.random.RandomState(seed)


@dataclasses.dataclass
class Distribution:
    state_name: str
    var_name: str
    n_bins: int
    distribution: str
    distribution_kwargs: Dict
    # patient_specific: bool = dataclasses.field(default=False)
    _static: bool = dataclasses.field(default=False)
    _distribution_callable: Callable = dataclasses.field(init=False, default=None)
    _bins: Dict = dataclasses.field(init=False, default=None)
    _bin_min: Dict = dataclasses.field(init=False, default=None)
    _bin_max: Dict = dataclasses.field(init=False, default=None)
    _random: np.random.RandomState = dataclasses.field(init=False, default=None)
    _static_val: str = dataclasses.field(init=False, default=None)

    def __post_init__(self):
        self._bins = {b: str(b) for b in range(self.n_bins)}
        self._bin_min = min(self._bins.keys())
        self._bin_max = max(self._bins.keys())
        self._random = str_to_state(f"{self.state_name}_{self.var_name}")
        self._distribution_callable = getattr(self, self.distribution)()
        # self._static = self.patient_specific
        if self.distribution_kwargs is None:
            self.distribution_kwargs = {}

    def _sample(self) -> str:
        if self._static:
            if self._static_val is None:
                self._static_val = self._distribution_callable(self.distribution_kwargs)
            return self._static_val
        # Not a static variable, so sample new event
        return self._distribution_callable(self.distribution_kwargs)

    def _round_and_clip(self, sample: float):
        _bin = np.round(sample).astype(int)
        return np.clip(_bin, self._bin_min, self._bin_max)

    def uniform(self) -> Callable:
        """Uniformly sample from one of the bins."""

        def fn(kwargs):
            sample = self._random.uniform(**kwargs)
            sample *= len(self._bins)
            _bin = self._round_and_clip(sample)
            return self._bins[_bin]

        return fn

    def hotspot_shuffle(self) -> Callable:
        """A hot-spot is a uniform distribution over a subset of the total bins, miniscule
        probability over the remaining events.

        hotspot_bins: The number of hotspot bins. Default 10
        hotspot_factor: The ratio of sum(hotspot_bins_logits):sum(non-hotspot_bins_logits). Default
        10
 
        """
        bin_keys = list(self._bins.keys())
        self._random.shuffle(bin_keys)
        _bin_choices = None

        def fn(kwargs):
            num_hotspot_bins = kwargs["num_hotspot_bins"]
            hotspot_factor = kwargs["hotspot_factor"]
            non_hotspot_prob = (
                (1 / (hotspot_factor + 1)) / 
                (len(self._bins) - num_hotspot_bins)
            )
            hotspot_prob = (
                (hotspot_factor / (hotspot_factor + 1)) / 
                (num_hotspot_bins)
            )
            probs = np.full(len(self._bins), non_hotspot_prob)
            probs[:num_hotspot_bins] = hotspot_prob
            # Iterative generation
            # _bin = self._random.choice(bin_keys, p=probs)
            # Batch generate 100 random choices at a time, much faster than 1-by-1
            nonlocal _bin_choices
            if _bin_choices is None or len(_bin_choices) == 0:
                _bin_choices = self._random.choice(bin_keys, 100, p=probs).tolist()
            _bin = _bin_choices.pop(0)
            return self._bins[_bin]

        return fn

    def normal(self) -> Callable:
        """Arrange the bins sequentially, and sample from them via a Gaussian
        distribution. Doesn't really make sense for categorical data,
        though.

        """

        def fn(kwargs):
            sample = self._random.normal(**kwargs)
            sample += (len(self._bins) // 2) + 0.5
            _bin = self._round_and_clip(sample)
            return self._bins[_bin]

        return fn

    def zipf_shuffle(self) -> Callable:
        """Shuffle the bins, then sample via a Zipf distribution.

        The uniqueness of the samples depends on the number of samples
        drawn, and the parameter `a`.  The more samples, the more
        likely there is to be a unique sample.  To give a vague idea
        of the uniqueness, here are the approx number of unique
        samples when drawing 100 samples:

        a = 2: unique ~ 13.3
        a = 3: unique ~  5.4
        a = 4: unique ~  3.3
        a = 5: unique ~  2.4
        a = 6: unique ~  2.0

        """
        bin_keys = list(self._bins.keys())
        support = len(bin_keys) + 1
        self._random.shuffle(bin_keys)
        pmf = None
        _bin_choices = None

        def fn(kwargs):
            nonlocal pmf
            if pmf is None:
                pmf = scipy.stats.zipfian.pmf(np.arange(support), kwargs["a"], support)

            nonlocal _bin_choices
            if _bin_choices is None or len(_bin_choices) == 0:
                _bin_choices = np.random.default_rng().choice(np.arange(len(pmf)), 100, p=pmf / pmf.sum()).tolist()
            sample = _bin_choices.pop(0)
            _bin = bin_keys[sample - 1]
            return self._bins[_bin]

            # sample = self._random.zipf(**kwargs)
            # if sample > len(bin_keys):
            #     # Sample out of range, try again
            #     return fn(kwargs)
            # _bin = bin_keys[sample - 1]
            return self._bins[_bin]

        return fn

    # def binomial_shuffle(self) -> Callable:
    #     """Shuffle the bins, then sample via a Binomial distribution.

    #     """
    #     bin_keys = list(self._bins.keys())
    #     self._random.shuffle(bin_keys)

    #     def fn(kwargs):
    #         sample = self._random.zipf(**kwargs)
    #         if sample > len(bin_keys):
    #             # Sample out of range, try again
    #             return fn(kwargs)
    #         _bin = bin_keys[sample - 1]
    #         return self._bins[_bin]

    #     return fn

@dataclasses.dataclass
class State:
    state_name: str
    distributions: Dict[str, Distribution]
    persistence_dict: Dict[str, float]
    state_children: List[str]
    _random: np.random.RandomState = dataclasses.field(init=False, default=None)
    _start: bool = dataclasses.field(default=False)
    _end: bool = dataclasses.field(default=False)

    def __post_init__(self):
        self._random = str_to_state(self.state_name)
        self._persistence = self._random.uniform(
            low=self.persistence_dict["min"],
            high=self.persistence_dict["max"],
            size=(1,),
        )[0]

        if len(self.state_children) == 0:
            self._end = True


        # self._start = self._random.random() < self.persistence_dict["start"]
        # self._end = self._random.random() < self.persistence_dict["end"]
        # if self._start and self._end:
        #     logging.warning("State: {self.state_name} is both a start and end state.")

    def _syn(self) -> Dict:
        row = {d.var_name: d._sample() for d in self.distributions.values()}
        row["state_id"] = self.state_name
        row["start"] = self._start
        row["end"] = self._end

        return row


@dataclasses.dataclass
class Patient:
    name: str
    states: List[State]
    _state_expression: State = dataclasses.field(init=False, default=None)
    _first_event: bool = dataclasses.field(init=False, default=True)
    _last_event: bool = dataclasses.field(init=False, default=False)
    # _distributions: Dict[str, Distribution] = dataclasses.field(
    #     init=False, default=None
    # )

    def __post_init__(self):
        self._distributions = {}
        if self._state_expression is None:
            self._sample_new_state()

    def _syn(self):
        if not self._first_event:
            self._state_change()
        else:
            # Don't change state on first event, since none have been sampled yet
            self._first_event = False
        return dict(
            **self._state_expression._syn(),
            # **{d.var_name: d._sample() for d in self._distributions.values()},
            patient_id=self.name,
        )

    def _state_change(self):
        """At any given time, a patients state can randomly change, and the
        probability of this change is dependent on the current state of the
        patient."""
        if GLOBAL_RANDOM_STATE.random() >= self._state_expression._persistence:
            if self._state_expression._end:
                # At this point in the control flow, we have no new state to advance to 
                # since this is an end state. Therefore, this is where synthesis should end.
                self._last_event = True
                return
            self._sample_new_state()

    def _sample_new_state(self):
        while True:
            if self._state_expression is None:
                # Sample the first states from states with no parents
                # This creates too much of a bias on the first state
                states = {state_name: s for state_name, s in self.states.items() if s._start}

                # Sample the first state from states which do have children (i.e., not end states)
                # states = {state_name: s for state_name, s in self.states.items() if not s._end}

                # Sample from any state, regardless of whether it is an end state or not
                # states = {state_name: s for state_name, s in self.states.items()}
            else:
                states = {
                    state_name: self.states[state_name] 
                    for state_name in self._state_expression.state_children
                }
            state = GLOBAL_RANDOM_STATE.choice(list(states.values()), 1)[0]
            if self._state_expression is None:
                break
            if state.state_name != self._state_expression.state_name:
                break
        self._state_expression = state


@dataclasses.dataclass
class Landscape:
    config_path: Union[Path, Dict]
    _config_yaml: Dict = dataclasses.field(init=False, default=None)
    _states: Dict[str, State] = dataclasses.field(init=False, default=None)
    _patients: List[Patient] = dataclasses.field(init=False, default=None)
    _events: int = dataclasses.field(init=False, default=None)

    def __post_init__(self):
        if isinstance(self.config_path, Path):
            with open(self.config_path, "r") as f:
                self._config_yaml = yaml.safe_load(f)            
        else:
            self._config_yaml = self.config_path
        # (0) Set global seed
        set_global_random_state(self._config_yaml["seed"])
        # (1) Parse the States
        self._states = self.state_generator(self._config_yaml)
        start_states = [s for s in self._states.values() if s._start]
        end_states = [s for s in self._states.values() if s._end]
        if len(start_states) == 0:
            raise ValueError(f"There are no start states! Try rolling a new seed.")
        if len(end_states) == 0:
            raise ValueError(f"There are no end states! Try rolling a new seed.")
        # (2) For each patient, assign it a starting-state from start_states
        self._patients = [
            Patient(
                name=PhonemicName().new_name(GLOBAL_RANDOM_STATE),
                states=self._states,
            )
            for _ in range(self._config_yaml["patients"])
        ]

    @property
    def n_states(self) -> int:
        return len(self._states)

    @property
    def n_patients(self) -> int:
        return len(self._patients)

    def variable_generator(self, cfg: Dict, state_name: str) -> List[Distribution]:
        var_names = [
            f"VAR_{str(i).zfill(len(str(cfg['nb_variables'])))}"
            for i in range(cfg["nb_variables"])
        ]
        return {
            var_name: Distribution(
                state_name=state_name,
                var_name=var_name,
                **cfg["variable_kwargs"],
            )
            for var_name in var_names
        }

    def state_generator(self, cfg: Dict) -> List[State]:
        state_cfg = cfg["states_generator"]
        variable_cfg = cfg["variable_generator"]

        self.G = ModelFactory.from_str(model=state_cfg["model"], model_kwargs=state_cfg["model_kwargs"])
        if len(self.G.nodes) != state_cfg["nb_states"]:  # Check that these are equal
            logging.warning(f"The number of nodes in the graph was not equal to the number of states. Forcing to number of states")
            model_kwargs = {**state_cfg["model_kwargs"], **{"n": state_cfg["nb_states"]}}
            self.G = ModelFactory.from_str(model=state_cfg["model"], model_kwargs=model_kwargs)

        state_names = [
            f"STATE_{str(i).zfill(len(str(state_cfg['nb_states'])))}"
            for i in range(state_cfg["nb_states"])
        ]
        # Determine start- and end-states
        # Start-states have only out-edges, while out-states have only in-edges (edge direction is reversed in nx)
        # If minimum values specificed in config are not met, then randomly add new ones until the specification is met
        state_children_map = {
            state_name: [state_names[idx] for idx in [t[0] for t in list(self.G.in_edges) if t[1] == n]]
            for state_name, n in zip(state_names, self.G.nodes)
        }
        state_start_map = {
            state_name: True 
            for state_name in state_names
            if not any(state_name in children for children in state_children_map.values())
        }
        state_end_map = {
            state_name: True 
            for state_name in state_names
            if (len(state_children_map[state_name]) == 0) and (not state_start_map.get(state_name, False)) 
        }
        while len(state_start_map) < state_cfg["min_start_states"]:
            # Make a random state a start state
            non_start_states = set(state_names) ^ (set(list(state_start_map)) | set(list(state_end_map)))
            state_start_map[GLOBAL_RANDOM_STATE.choice(list(non_start_states))] = True
        while len(state_end_map) < state_cfg["min_end_states"]:
            # Make a random state an end state
            non_end_states = set(state_names) ^ (set(list(state_start_map)) | set(list(state_end_map)))
            state_end_map[GLOBAL_RANDOM_STATE.choice(list(non_end_states))] = True
        return {
            state_name: State(
                state_name=state_name,
                state_children=state_children_map[state_name],
                distributions=self.variable_generator(variable_cfg, state_name),
                _start=state_start_map.get(state_name, False),
                _end=state_end_map.get(state_name, False),
                **state_cfg["state_kwargs"],
            )
            for state_name in state_names
        }

    def draw_graph(self):
        # Draw the graph using graph-tool, because it makes prettier graphs.
        gtG = nx2gt(self.G)
        # Colour the nodes according to their start/end node status
        cmap = cm.get_cmap("Pastel1")
        cmap = {
            "r": (*cmap.colors[0], 1.),
            "g": (*cmap.colors[2], 1.),
            "b": (*cmap.colors[1], 1.),
            "y": (*cmap.colors[5], 1.),
        }
        # Create new vertex property
        plot_color = gtG.new_vertex_property('vector<double>')
        # add that property to graph
        gtG.vertex_properties['plot_color'] = plot_color
        # assign a value to that property for each node of that graph
        for v, (state_name, state) in zip(gtG.vertices(), self._states.items()):
            in_e, out_e = list(v.in_edges()), list(v.out_edges())
            if len(in_e) == 0 and len(out_e) != 0:
                # This is a start-node
                plot_color[v] = cmap["g"]
            if len(in_e) != 0 and len(out_e) == 0:
                # This is an end-node
                plot_color[v] = cmap["r"]
            if len(in_e) != 0 and len(out_e) != 0:
                # These are intermediate nodes
                plot_color[v] = cmap["b"]
            if len(in_e) == 0 and len(out_e) == 0:
                # These are isolated nodes
                plot_color[v] = cmap["y"]

        gt_graph_draw(
            gtG, 
            vertex_text=gtG.vertex_index,
            output=str(self.config_path.parent / 'graph.png'), 
            bg_color='white',
            vertex_fill_color=gtG.vertex_properties['plot_color'],
            fit_view=True,
            output_size=(600, 600),
            adjust_aspect=False,
        )
        gt_graph_draw(
            gtG, 
            vertex_text=gtG.vertex_index,
            output=str(self.config_path.parent / 'graph_alpha.png'), 
            bg_color=None,
            vertex_fill_color=gtG.vertex_properties['plot_color'],
            fit_view=True,
            output_size=(600, 600),
            adjust_aspect=False,
        )
        gt_graph_draw(
            gtG, 
            vertex_text=gtG.vertex_index,
            output=str(self.config_path.parent / 'graph.pdf'), 
            bg_color=None,
            vertex_fill_color=gtG.vertex_properties['plot_color'],
            fit_view=True,
            output_size=(600, 600),
            adjust_aspect=False,
        )

    def save_adjacenct_matrix(self):
        am = nx.adjacency_matrix(self.G).toarray()
        np.savez_compressed(self.config_path.parent / "syn_G_adjacency_matrix.npz", am=am)

    def syn(self) -> pd.DataFrame:
        df = []
        for patient in self._patients:
            num_events = 0
            while not patient._last_event:
                if num_events == MAX_NUM_EVENTS:
                    break
                df.append(patient._syn())
                num_events += 1
        
        return pd.DataFrame(df)
