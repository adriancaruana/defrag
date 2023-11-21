import pandas as pd
from pathlib import Path
import yaml
import numpy as np
from tqdm import tqdm

from _constants import (
    # Seq2Seq
    SEQ2SEQ_MODEL,
    SEQ2SEQ_LOSS_PLOT,
    SEQ2SEQ_BATLOW_LOSS_PLOT,
    SEQ2SEQ_ENCODING_IMG,
    SEQ2SEQ_DECODING_IMG,
    SEQ2SEQ_CLUSTERING_IMG,
    FINAL_REPRESENTATIONS,
    # Defrag
    CLUSTERS_CSV,
    CLUSTERS_STATS,
    NXG_PLOT,
    GTG_PLOT,
    GTG_PLOT_ALPHA,
    SOFT_ADJACENCY_MATRIX,
)


def read_config(experiment: Path):
    with open(next(experiment.glob("catsyn_config*.yml")), 'r') as f:
        return yaml.safe_load(f)

def read_clustered_stats(experiment: Path):
    stats = list(experiment.glob("clustered_stats.yaml"))
    if len(stats) == 0:
        return {}
    with open(stats[0], 'r') as f:
        return yaml.safe_load(f)

def read_df(experiment: Path):
    df_path = list(experiment.glob("syn*.csv"))[0]
    return pd.read_csv(df_path)

def read_defrag_results(experiment: Path):
    path = experiment / "defrag_results.yaml"
    if not path.exists():
        return {}
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def gen_dict_extract(key, var):
    if hasattr(var,'items'):
        for k, v in var.items():
            if k == key:
                yield v
            if isinstance(v, dict):
                for result in gen_dict_extract(key, v):
                    yield result
            elif isinstance(v, list):
                for d in v:
                    for result in gen_dict_extract(key, d):
                        yield result

def change_link_root(link):
    return "/".join(str(link.absolute()).split("/")[5:])

def make_clickable(val, change_root: bool = False):
    """Makes links in DataFrames clickable."""
    # target _blank to open new window
    if val[1] == "n/a":
        return ""
    link = val[0]
    if change_root:
        link = change_link_root(val[0])
    return '<a target="_blank" href="{}">{}</a>'.format(link, val[1])

def make_numeric(val):
    if val is None or np.isnan(val) or val == -1:
        return ""
    if float(val).is_integer():
        return str(int(val))
    return f"{round(val * 100) / 100}"

def R(s, c):
    if c == -1:
        return ""
    return {0: "🟩", 1: "🟨", 2: "🟧",}.get(abs(s - c), "🟥")

def iso(v):
    if v is None or v == "":
        return ""
    return "🟩" if v else "🟥"

def compute_GED_metric(actual_ged, num_states):
    if actual_ged is None:
        return None
    return actual_ged / num_states

def view_results(results_root: Path, raw=False, change_root=False) -> pd.DataFrame:
    if "mimic" in str(results_root):
        return
    experiments = [
        {
            "path": Path(exp_path),
            "catsyn_config": read_config(exp_path),
            "clustered_stats": read_clustered_stats(exp_path),
            "df": read_df(exp_path),
            "defrag_results": read_defrag_results(exp_path),
        }
        for exp_path in results_root.glob("experiment*")
    ]

    experiment_df = []
    for e in tqdm(experiments):
        experiment_df.append({
            "name": e['path'].name,
            "seed": e['catsyn_config']['seed'],
            "#P": e['catsyn_config']['patients'],
            "#V": e['catsyn_config']['variable_generator']['nb_variables'],
            "zipf_a": e['catsyn_config']['variable_generator']['variable_kwargs']['distribution_kwargs']['a'],
            "#E": f"{round(len(e['df']) / 1000)}k",
            "#S": e['catsyn_config']['states_generator']['nb_states'],
            "#B": e['catsyn_config']['variable_generator']['variable_kwargs']['n_bins'],
            "G": (None, "n/a") if not (path := (e['path'] / "graph.png")).exists() else (path, "graph.png"[-3:]),
            "loss": (None, "n/a") if not (path := (e['path'] / SEQ2SEQ_LOSS_PLOT)).exists() else (path, SEQ2SEQ_LOSS_PLOT[-3:]),
            "UMAP E": (None, "n/a") if not (path := (e['path'] / SEQ2SEQ_ENCODING_IMG)).exists() else (path, SEQ2SEQ_ENCODING_IMG[-3:]),
            "UMAP D": (None, "n/a") if not (path := (e['path'] / SEQ2SEQ_DECODING_IMG)).exists() else (path, SEQ2SEQ_DECODING_IMG[-3:]),
            "#C": e.get("clustered_stats", {}).get('nb_clusters', -1),
            "AMI": e.get("clustered_stats", {}).get('ami', None),
            "MS": (di := e.get("clustered_stats", {}).get('clusterer_info', {})).get('min_samples', di.get("params", {}).get('min_samples', None)),
            "MCS": (di := e.get("clustered_stats", {}).get('clusterer_info', {})).get('min_cluster_size', di.get("params", {}).get('min_cluster_size', None)),
            "UMAP C": (None, "n/a") if not (path := (e['path'] / SEQ2SEQ_CLUSTERING_IMG)).exists() else (path, SEQ2SEQ_CLUSTERING_IMG[-3:]),
            "SAM": (None, "n/a") if not (path := (e['path'] / SOFT_ADJACENCY_MATRIX)).exists() else (path, SOFT_ADJACENCY_MATRIX[-3:]),
            "NXG": (None, "n/a") if not (path := (e['path'] / NXG_PLOT)).exists() else (path, NXG_PLOT[-3:]),
            "GTG": (None, "n/a") if not (path := (e['path'] / GTG_PLOT)).exists() else (path, GTG_PLOT[-3:]),
            "ISO": iso(e.get('defrag_results', {}).get("is_isomorphic", "")),
            "S⊑I": iso(e.get('defrag_results', {}).get("inf_contains_syn", "")),
            "I⊑S": iso(e.get('defrag_results', {}).get("syn_contains_inf", "")),
            "GED": compute_GED_metric(e.get('defrag_results', {}).get("edit_distance", None), e['catsyn_config']['states_generator']['nb_states'])
        })

    clickable_cols = ["G", "loss", "UMAP E", "UMAP D", "UMAP C", "SAM", "NXG", "GTG"]
    numeric_cols = ["#C", "AMI", "MS", "MCS", "GED"]

    df = pd.DataFrame(experiment_df).sort_values(["#P", "#S", "zipf_a"], ascending=[True, True, False])
    df["R"] = df.apply(lambda x: R(x['#S'], x["#C"]), axis=1)
    if raw:
        return df    
    # df = df[df["name"].apply(lambda x: "Clo+Sep+Con+MSE" in x)]
    df = df.style.format({**{col: lambda x: make_clickable(x, change_root=change_root) for col in clickable_cols}, **{col: make_numeric for col in numeric_cols}})
    return df
