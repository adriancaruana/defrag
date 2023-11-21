from pathlib import Path
import sys
import shutil
from itertools import product
from typing import List, Dict
import logging
import yaml

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats
from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.append("../")
from catsyn.catsyn import syn, ephemeral_syn
from catsyn_config_generator import CatsynConfigFeedstock
from conductor import run_synthesis

from tqdm_joblib import tqdm_joblib

reference_data_infos = np.load("dataset_info.npz", allow_pickle=True)["data"].item(0)
for ref_name, ref_dict in reference_data_infos.items():
    ref_dict["ref_f"] = ref_dict["ref_f"].tolist()

# def run_experiment(config):
#     root = Path("./experiments")
#     config = next(config.permute_configs())
#     # config = config
#     experiment_path = root / config.experiment_name
#     if experiment_path.exists():
#         shutil.rmtree(experiment_path)
#     experiment_path.mkdir(parents=True, exist_ok=True)
#     config.generate_and_save_config(experiment_path)
#     # Generate the synthetic data
#     # Read in the resulting data
#     try:
#         print(f"{config=}")
#         run_synthesis(config, experiment_path=experiment_path, save=True)
#         df = get_syn_df(experiment_path)
#     except:
#         raise RuntimeError(f"Experiment {experiment_path} failed")

#     # if experiment_path.exists():
#     #     shutil.rmtree(experiment_path)
#     return df

def run_experiment(config):
    df = ephemeral_syn(
        config,
    )
    return df



def get_syn_df(experiment_path: Path) -> np.array:
    csv_path = next(experiment_path.glob("*.csv"))
    return pd.read_csv(csv_path)


def get_config_from_dist(dist, dist_kwargs, graph = None):
    if dist == "zipf_shuffle":
        DISTRIBUTION = [("zipf_shuffle", {"a": float(dist_kwargs["a"])})]
    elif dist == "normal":
        DISTRIBUTION=[("normal", {"scale": float(dist_kwargs["scale"])}),]
    elif dist == "hotspot_shuffle":
        DISTRIBUTION=[("hotspot_shuffle", {
            "num_hotspot_bins": int(dist_kwargs["num_hotspot_bins"]), 
            "hotspot_factor": float(dist_kwargs["hotspot_factor"])
        }),]
    elif dist == "uniform":
        DISTRIBUTION=[("uniform", {}),]
    else:
        raise ValueError()

    model = [("directed_extended_barabasi_albert_graph", None),]
    if graph is not None:
        if graph == "gnp_random_graph":
            p = np.sqrt(int(dist_kwargs["states"])) / int(dist_kwargs["states"]) * float(dist_kwargs["p"])
            model = [(graph, {"n": int(dist_kwargs["states"]), "p": float(p)}),]
        elif graph == "directed_extended_barabasi_albert_graph":
            if int(dist_kwargs["m"]) >= int(dist_kwargs["states"]):
                return None
            if float(dist_kwargs["p"]) + float(dist_kwargs["q"]) >= 1:
                return None
            model = [(graph, {
                "n": int(dist_kwargs["states"]), 
                "m":  int(dist_kwargs["m"]), 
                "p": float(dist_kwargs["p"]), 
                "q": float(dist_kwargs["q"])
            }),]
        elif graph == "binomial_tree":
            model = [(graph, {"n": int(np.log2(int(dist_kwargs["states"])))}),]

    config = CatsynConfigFeedstock(
        NB_PATIENTS=int(dist_kwargs["patients"]),
        NB_STATES=int(dist_kwargs["states"]),
        NB_VARIABLES=1,
        NB_BINS=int(dist_kwargs["bins"]),
        DISTRIBUTION=DISTRIBUTION,
        MODEL=model,
        MIN_START_STATES=1,
        MIN_END_STATES=1,
        PERSISTENCE_MIN=float(dist_kwargs["persistence_min"]),
        PERSISTENCE_MAX=float(dist_kwargs["persistence_max"]),
        SEED=int(dist_kwargs.get("persistence_max", 24))
    )
    return config


def fix_lens(f1, f2):
    if len(f1) < len(f2):
        append_vals = np.zeros((len(f2) - len(f1))) + 1e-10
        f1 = np.append(f1, append_vals)

    if len(f2) < len(f1):
        append_vals = np.zeros((len(f1) - len(f2))) + 1e-10
        f2 = np.append(f2, append_vals)

    assert len(f2) == len(f1)
    return f1, f2


def get_metrics(reference_info, syn_df, syn_kwargs, syn_dist, plot):
    syn_var = "VAR_0"
    syn_f = syn_df[syn_var].value_counts().values
    syn_f = syn_f / syn_f.sum()
    synthetic_data_info = dict(
        syn_name=syn_dist,
        syn_num_bins=syn_df[syn_var].unique().__len__(),
        syn_num_patients=syn_df["patient_id"].unique().__len__(),
        syn_num_events=syn_df.__len__(),
        syn_avg_seq_len=syn_df.__len__()/syn_df["patient_id"].unique().__len__(),
        syn_f=syn_f,
        # avg_unique_events_per_patient=np.mean([df[df.patient_id.isin([pid])].p_icd9_code.unique().__len__() for pid in tqdm(df.patient_id.unique())]),
    )

    metrics = []
    for ref_info in reference_info.values():
        _ref_f = ref_info["ref_f"].copy()
        _syn_f = syn_f.copy()
        _ref_f, _syn_f = fix_lens(_ref_f, _syn_f)

        P = _ref_f
        Q = _syn_f
        entropy = scipy.stats.entropy(P, Q)

        if plot:
            plt.plot(np.arange(len(_ref_f)), _ref_f, color='r', alpha=0.5, label=ref_info["ref_name"])
            plt.plot(np.arange(len(_syn_f)), _syn_f, color='b', alpha=0.5, label=f"synthetic ({syn_dist})")
            plt.title(f"Frequencies of reference and synthetic\nentropy={entropy:.3f}")
            plt.legend()
            plt.show()
        comparison_info = dict(
            entropy=entropy
        )
        metrics.append({
            **ref_info,
            **synthetic_data_info,
            **syn_kwargs,
            **comparison_info,
        })
    return metrics


def fn(syn_dist: str, syn_kwargs: dict, reference_data_infos, plot=False, graph=None, return_entropy=True):
    
    config = get_config_from_dist(syn_dist, syn_kwargs, graph=graph)
    if config == None:
        return None
    config = next(config.permute_configs()).to_config()

    syn_df = run_experiment(config)
    metrics = get_metrics(reference_data_infos, syn_df, syn_kwargs, syn_dist, plot=plot)
    # print{patients=}, {states=}, {bins=}, {zipf_a=}, {entropy=}")
    return metrics


if __name__ == "__main__":
    if Path("/workspaces/defrag/statistical_experiments/experiments").exists():
        shutil.rmtree("/workspaces/defrag/statistical_experiments/experiments")
    else:
        Path("/workspaces/defrag/statistical_experiments/experiments").mkdir(exist_ok=True)

    grid = {
        # "patients": np.concatenate([np.arange(1_000, 10_000, 1500)][::-1]), 
        # "patients": [1000, 2500, 5000, 10000], 
        "patients": [5000], 
        # "states": list(range(100, 200, 25))[::-1] + list(range(10, 100, 20))[::-1], 
        "states": list([2, 4, 8, 16, 32, 64])[::-1], 
        # "states": [2, 5, 10, 15, 20, 30], 
        # "bins": np.concatenate([np.arange(250, 3000, 250)]), 
        # "bins": [1000, 2500, 5000, 10000], 
        "bins": [10000], 
        # For Zipf
        "a": np.arange(1.01, 2.1, 0.2),

        # For Normal
        # "scale": np.concatenate([np.arange(0.1, 1, 0.1), np.arange(2, 3, 0.25), np.arange(3, 10, 0.5)]),

        # For Hotspot
        # "num_hotspot_bins": [1, 5, 10, 20, 50],
        # "hotspot_factor": [1, 5, 10, 20, 50],

        # For GNP
        # "p": np.arange(0.1, 1, 0.1),  # Probability of edge

        # For Barabasi
        # "m": np.arange(1, 11, 2),  # Number of edges to attach from a new node to existing nodes
        # "p": np.arange(0.0, 1, 0.2),  # Probability of edge
        # "q": np.arange(0.0, 1, 0.2),  # Probability of edge 



        "persistence_min": [0.7],
        "persistence_max": [0.7],
    }

    dist = "zipf_shuffle"
    # graph = None
    # graph = "directed_extended_barabasi_albert_graph"
    # graph = "gnp_random_graph"
    graph = "binomial_tree"

    param_list = [dict(zip(grid, v)) for v in product(*grid.values())]

    # param_list = param_list[:10]
    # fn(syn_dist=dist, syn_kwargs=param_list[0], reference_data_infos=reference_data_infos)

    with tqdm_joblib(tqdm(total=len(param_list), bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')) as progress_bar:
        metrics: List[List[Dict]] = Parallel(n_jobs=12)(
            delayed(fn)(syn_dist=dist, syn_kwargs=params, reference_data_infos=reference_data_infos, graph=graph)
            for params in param_list
        )

    # Unroll the metrics
    metrics = [m for m in metrics if m is not None]
    metrics = [mm for m in metrics for mm in m]
    # Save results
    df = pd.DataFrame(metrics)
    del df["ref_f"]
    Path("result_datasets").mkdir(exist_ok=True)
    df.to_parquet(f"result_datasets/grid_search_results_{dist}_{graph}.parquet")