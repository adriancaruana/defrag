from pathlib import Path
import logging
import shutil
import sys
from typing import Dict, Optional, Union
import dataclasses
import gc

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import yaml

# catsyn imports
from catsyn_config_generator import CatsynConfigFeedstock
from catsyn.catsyn import syn

# CaSE imports
from case_config_generator import Seq2SeqConfigFeedstock
from case.case import Experiment, Seq2SeqConfig, load_model, save_model
from case.data import C2VDataGen

# Defrag imports
from defrag import DefragConfig, Defrag, Eval
from mimic_config_generator import MimicConfigFeedstock

# notifications
from notify import notify

import experiment_configs
from filter_mimic import filter_mimic_features

from _constants import (
    # Experiment artefacts
    # Seq2Seq
    SEQ2SEQ_CONFIG_NAME,
    SEQ2SEQ_MODEL,
    SEQ2SEQ_LOSS_PLOT,
    SEQ2SEQ_BATLOW_LOSS_PLOT,
    SEQ2SEQ_ENCODING_IMG,
    SEQ2SEQ_DECODING_IMG,
    SEQ2SEQ_CLUSTERING_IMG,
    FINAL_REPRESENTATIONS,
    # Defrag
    DEFRAG_CONFIG_NAME,
    CLUSTERS_CSV,
    CLUSTERS_STATS,
    NXG_PLOT,
    GTG_PLOT,
    GTG_PLOT_ALPHA,
    GTG_PLOT_PDF,
    GTG_SOFT_PLOT,
    GTG_SOFT_PLOT_ALPHA,
    GTG_SOFT_PLOT_PDF,
    SOFT_ADJACENCY_MATRIX,
    DEFRAG_RESULTS,
    DEFRAG_DATA,
)

logging.basicConfig()
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


DEVICE = None


CONTINUE_ON_ERROR = True

CATSYN_FEEDSTOCK = CatsynConfigFeedstock(
    NB_PATIENTS=[500],
    NB_STATES=[12],#, 3, 6],
    NB_VARIABLES=[3],
    # NB_BINS=[100],
    DISTRIBUTION=[
        # ("zipf_shuffle", {"a": 2.5}),
        # ("zipf_shuffle", {"a": 2}),
        ("zipf_shuffle", {"a": 3}),
        # ("zipf_shuffle", {"a": 4}),
    ],
    MODEL=[
        ("directed_extended_barabasi_albert_graph", None),
        # ("gnp_random_graph", {"n": 50, "p": 0.2}),
    ],
)


MIMIC_FEEDSTOCK = MimicConfigFeedstock(
    # "2.1",  # Colorectal cancer
    # "2.2",  # Skin cancer
    # "2.5",  # Breast cancer
    # "9.6.1" # Appendicitis and other appendiceal conditions

    CANCER_CCS_CODE={2: "2.5"},# 3: "9.6.1"},
    MIN_SEQ_LEN=16,
    TFIDF_DOC_COL="patient_id",
    TFIDF_VOCAB_COL="p_icd9_code",
    DIAGNOSIS_SELECTION="full",
    COLS=(
        # "hadm_idx", 
        # "admission_type", "admission_location",  
        #"d_icd9_code", "d_ccs_lv1", "d_ccs_lv2", "d_ccs_lv3",
        "p_icd9_code", "p_ccs_lv1", "p_ccs_lv2", "p_ccs_lv3",
    ),
    DIAG_ONLY=False,
    PROC_ONLY=True,
)


LOGGER.info("Starting")

def get_results_path():
    assert len(sys.argv[1:]) >= 1, "No path for results directory specified."
    ROOT = Path(__file__).absolute().parent

    if is_adhoc_experiment():
        RESULTS_PATH = ROOT / sys.argv[1]
        return RESULTS_PATH

    experiment_name, _ = get_experiment()
    RESULTS_PATH = ROOT / "experiments" / experiment_name
    return RESULTS_PATH


def is_mimic_experiment():
    if len(sys.argv[1:]) >= 2 and sys.argv[2] == "mimic":
        return True
    return False

def is_adhoc_experiment():
    if len(sys.argv[1:]) >= 2 and sys.argv[2] == "adhoc":
        return True
    return False

def get_experiment():
    assert len(sys.argv[1:]) == 1
    experiment_name = sys.argv[1]
    assert hasattr(experiment_configs, experiment_name), f"No experiment named {experiment_name=}. Maybe check the spelling?"
    return experiment_name, getattr(experiment_configs, experiment_name)
    

def init_device():
    global DEVICE
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Torch device: {DEVICE}")


# def clean_results():
#     for dir in filter(lambda x: x.is_dir(), RESULTS_PATH.glob("*")):
#         shutil.rmtree(dir)


def clean_clustering(experiment: Path):
    files = []
    # Re-do Defrag
    # files += [
    #     NXG_PLOT,
    #     GTG_PLOT,
    #     GTG_PLOT_ALPHA,
    #     SOFT_ADJACENCY_MATRIX,
    #     DEFRAG_RESULTS
    # ]
    # # Re-do clustering
    # files += [
    #     SEQ2SEQ_CLUSTERING_IMG,
    #     CLUSTERS_CSV,
    #     CLUSTERS_STATS,
    # ]
    # # Re-do encoding
    # files += [
    #     SEQ2SEQ_ENCODING_IMG,
    #     SEQ2SEQ_DECODING_IMG,
    #     FINAL_REPRESENTATIONS,
    # ]
    # # Re-do training
    # files += [
    #     SEQ2SEQ_MODEL,
    #     SEQ2SEQ_LOSS_PLOT,
    # ]
    for file in files:
        if (f := (experiment / file)).exists():
            LOGGER.warning(f"Removing file: {f}.")
            f.unlink()


def _syn(
    config: Path,
    save_dir: Path,
    save_graph: Optional[bool] = False,
    force: Optional[bool] = False,
) -> Path:
    try:
        dataset_path = syn(
            config,
            save_dir,
            save_graph=True,
            force=True,
        )
    except Exception as e:
        if "Found dataset at" in str(e):
            LOGGER.warning(str(e))
        else:
            raise e
    return dataset_path


def run_synthesis(catsyn_config: CatsynConfigFeedstock, min_sequence_length: int = -1, experiment_path: Path = None, save=True) -> Dict:
    experiment_name = catsyn_config.experiment_name
    if experiment_path is None:
        experiment_path = get_results_path() / experiment_name
    experiment_path.mkdir(parents=True, exist_ok=True)
    synthetic_data_config_path = experiment_path / catsyn_config.config_name
    synthetic_dataset_save_dir = experiment_path
    # Generate the synthetic data config for this permutation
    if save:
        catsyn_config.generate_and_save_config(experiment_path)
    # Generate the synthetic dataset
    LOGGER.info("Generating synthetic data.")
    dataset_path = _syn(
        synthetic_data_config_path,
        synthetic_dataset_save_dir,
        save_graph=save,
        force=True,
    )
    # Remove short sequences from the dataset
    _synthetic_df = pd.read_csv(dataset_path)
    vc = _synthetic_df.patient_id.value_counts()
    patient_ids_seq_too_short = vc.index[vc >= min_sequence_length].tolist()
    if len(patient_ids_seq_too_short) > 0:
        LOGGER.info(f"Removing {len(_synthetic_df.patient_id.unique()) - len(patient_ids_seq_too_short)} patients from synthetic data since they're too short.")
    _synthetic_df = _synthetic_df[_synthetic_df.patient_id.isin(patient_ids_seq_too_short)]
    _synthetic_df.to_csv(dataset_path, index=False)
    return {"experiment_name": experiment_name, "dataset_path": dataset_path}


def run_mimic_filtering(mimic_config: MimicConfigFeedstock, min_sequence_length: int) -> Dict:
    experiment_name = mimic_config.experiment_name
    experiment_path = get_results_path() / experiment_name
    experiment_path.mkdir(parents=True, exist_ok=True)

    mimic_config_path = experiment_path / mimic_config.config_name
    dataset_path = experiment_path / "filtered_mimic_features.parquet"
    mimic_config.generate_and_save_config(experiment_path)

    config_dict = mimic_config.to_config()
    config_dict["min_seq_len"] = min_sequence_length
    df = filter_mimic_features(**config_dict)
    df.to_parquet(dataset_path)

    return {"experiment_name": experiment_name, "dataset_path": dataset_path}


def run_seq2seq(
    exp: Experiment,
    seq2seq_config: Seq2SeqConfig,
):
    dg = C2VDataGen(exp.data["full"]["data"], seq2seq_config.cols)
    i_shape_dict = dg.i_shape()
    encoding_lut = dg._encoding_lut
    exp_path = exp.data_path.parent
    with open(exp_path / SEQ2SEQ_CONFIG_NAME, "w") as f:
        f.write(yaml.dump(dataclasses.asdict(seq2seq_config)))
    # load/train the model
    if (seq2seq_model_path := (exp_path / SEQ2SEQ_MODEL)).exists():
        LOGGER.warning("Loading the model found at {seq2seq_model_path}.")
        seq2seq_model = exp._build_seq2seq_model(seq2seq_config, i_shape_dict)
        seq2seq_model = load_model(seq2seq_model, seq2seq_model_path)
    else:
        LOGGER.warning("Training the seq2seq model.")
        seq2seq_model = exp.train_seq2seq(
            dg,
            seq2seq_config,
            plot_loss=(exp_path / SEQ2SEQ_LOSS_PLOT),
            plot_barlow_loss=(exp_path / SEQ2SEQ_BATLOW_LOSS_PLOT),
        )
        save_model(seq2seq_model, seq2seq_model_path)

    conditional_encoding_kwargs = {
        "test": {
            "noise": None,
            "umap_kwargs": dict(n_neighbors=300, min_dist=0.0, n_components=2),
            "mode": "all",
            "join": False,
        },
    }
    plot = True
    # if seq2seq_encoding_img_path.exists():
    #     plot = False
    CPU = torch.device("cpu")
    seq2seq_model.to(CPU)
    seq2seq_model.eval()
    seq2seq_results = exp.encode_seq2seq(
        cat2vec_datagen=dg,
        seq2seq_model=seq2seq_model,
        seq2seq_config=seq2seq_config,
        on="encodings",
        dataset="full",
        plot=plot,
        plot_save_path=exp_path / SEQ2SEQ_ENCODING_IMG,
        **conditional_encoding_kwargs["test"],
    )
    seq2seq_decodings = exp.encode_seq2seq(
        cat2vec_datagen=dg,
        seq2seq_model=seq2seq_model,
        seq2seq_config=seq2seq_config,
        on="decodings",
        dataset="full",
        plot=plot,
        plot_save_path=exp_path / SEQ2SEQ_DECODING_IMG,
        **conditional_encoding_kwargs["test"],
    )
    seq2seq_results['decodings'] = seq2seq_decodings['encodings']
    seq2seq_results['decodings_embeddings'] = seq2seq_decodings['embeddings']    
    return {
        "seq2seq_model": seq2seq_model,
        "seq2seq_results": seq2seq_results,
    }


def defrag(
    encodings: Dict,
    case_experiment: Experiment,
    defrag_config: DefragConfig,
    cluster_on: str = "encodings",
    plot_on: str = "embeddings",
    # cluster_on: str = "decodings",
    # plot_on: str = "decodings_embeddings",
):
    exp_path = case_experiment.data_path.parent
    with open(exp_path / DEFRAG_CONFIG_NAME, "w") as f:
        f.write(yaml.dump(dataclasses.asdict(defrag_config)))
    np.savez_compressed(exp_path / FINAL_REPRESENTATIONS, **encodings)

    # (1) Cluster the encodings
    clusters_csv_path = exp_path / CLUSTERS_CSV
    clusters_stats_path = exp_path / CLUSTERS_STATS
    defrag = Defrag(
        encodings=encodings,
        case_experiment=case_experiment,
        cluster_on=cluster_on,
        **dataclasses.asdict(defrag_config),
    )
    if not (clusters_csv_path.exists() and clusters_stats_path.exists()):
        LOGGER.warning(f"Running Clustering on: {cluster_on=}")
        clusters_and_scores = defrag.cluster_and_score(
            classify_noise=True,
            plot_save_path=exp_path / SEQ2SEQ_CLUSTERING_IMG,
            plot_on=plot_on,
        )
        cluster_df = defrag.clusters_to_df(clusters_and_scores)
        cluster_df.to_csv(clusters_csv_path, index=False)
        stats = defrag.get_clustered_stats(clusters_and_scores)
        with open(clusters_stats_path, "w") as f:
            yaml.dump(stats, f)
    else:
        LOGGER.warning(f"Clustering already finished. Loading from {clusters_csv_path}")
        cluster_df = pd.read_csv(clusters_csv_path)

    # (2) Get a sequence of clusters for each patient
    pw_est_targets = {}
    marker = 0
    pw_seq_embeddings = encodings["pw_seq_embeddings"]
    if (
        e_len := sum(len(p_seq["targets"]) for p_seq in pw_seq_embeddings.values())
    ) != len(cluster_df):
        raise ValueError(
            "Expected encodings and cluster to have the same number of events."
            f"Got: encoding length: {e_len}, cluster length: {len(cluster_df)}"
        )
    for patient_id, patient_dict in pw_seq_embeddings.items():
        p_len = patient_dict["embeddings"].shape[0]
        pw_est_targets[patient_id] = cluster_df["y_hat"][marker : marker + p_len]
        marker += p_len

    # (3) Get and transform adjacency matrices: Raw -> Soft -> Hard
    ram = defrag.get_raw_adjacency_matrix(
        pw_est_targets, cluster_df["y_hat"].unique().tolist()
    )
    sam = defrag.get_soft_adjacency_matrix(ram)
    ham = defrag.get_hard_adjacenct_matrix(sam, threshold=0.2)

    # (4) Convert to nxG and generate plots
    LOGGER.info("Saving defrag results.")
    LOGGER.info("Creating and plotting soft graph.")
    np.savez_compressed(exp_path / DEFRAG_DATA, sam=sam, ham=ham, ram=ram)
    soft_nxG = defrag.am_to_nxG(sam)
    defrag.plot_dag_nx(soft_nxG, plot=(exp_path / NXG_PLOT))
    defrag.plot_adjacency_matrix(
        sam, tick_labels=list(soft_nxG.nodes()), plot=(exp_path / SOFT_ADJACENCY_MATRIX)
    )
    LOGGER.info("Creating and plotting hard graph.")
    hard_nxG = defrag.am_to_nxG(ham)
    soft_nxG = defrag.am_to_nxG(sam)
    defrag.plot_gtG_from_nxG(hard_nxG, plot=(exp_path / GTG_PLOT), colour_by=defrag_config.colour_by)
    defrag.plot_gtG_from_nxG(hard_nxG, plot=(exp_path / GTG_PLOT_ALPHA), bg_colour=None, colour_by=defrag_config.colour_by)
    defrag.plot_gtG_from_nxG(hard_nxG, plot=(exp_path / GTG_PLOT_PDF), bg_colour=None, colour_by=defrag_config.colour_by)
    defrag.plot_gtG_from_nxG(soft_nxG, plot=(exp_path / GTG_SOFT_PLOT), colour_by=defrag_config.colour_by)
    defrag.plot_gtG_from_nxG(soft_nxG, plot=(exp_path / GTG_SOFT_PLOT_ALPHA), bg_colour=None, colour_by=defrag_config.colour_by)
    defrag.plot_gtG_from_nxG(soft_nxG, plot=(exp_path / GTG_SOFT_PLOT_PDF), bg_colour=None, colour_by=defrag_config.colour_by)

    # (5) Score the inferred graph
    # Mimic has no ground truth, so skip for mimic.
    # We can make this part of post-processing the experiments
    # if not is_mimic_experiment():
    #     LOGGER.info("Evaluating the inferred graph.")
    #     syn_am = np.load(exp_path / "syn_G_adjacency_matrix.npz")['am']
    #     syn_G_nx = nx.from_numpy_matrix(syn_am.T, create_using=nx.DiGraph)
    #     inf_G_nx = nx.from_numpy_matrix(ham, create_using=nx.DiGraph)
    #     defrag_results = Eval(syn_G_nx, inf_G_nx).eval()
    #     defrag_results_path = exp_path / DEFRAG_RESULTS
    #     with open(defrag_results_path, "w") as f:
    #         yaml.dump(defrag_results, f)

    LOGGER.info("All done, nothing left to do!")


def run_permutation(
    data_config: Union[CatsynConfigFeedstock, MimicConfigFeedstock],
    seq2seq_config: Seq2SeqConfig,
    defrag_config: DefragConfig,
):
    min_sequence_length = (
        1 +  # For no nans in attention
        2 +  # For standard deviation (loss)
        1 +  # For maximum and remaining distances (loss)
        1  # For adjacent Distances (loss)
    )
    if isinstance(data_config, MimicConfigFeedstock):
        min_sequence_length = data_config.MIN_SEQ_LEN
        data_results = run_mimic_filtering(data_config, min_sequence_length=min_sequence_length)
    if isinstance(data_config, CatsynConfigFeedstock):
        data_results = run_synthesis(data_config, min_sequence_length=min_sequence_length)
    # CaSE Experiment
    exp = Experiment(data_results["dataset_path"])
    exp_path = exp.data_path.parent
    clean_clustering(exp_path)

    if not (npz := (exp_path / FINAL_REPRESENTATIONS)).exists():
        # Train Seq2Seq
        seq2seq_results = run_seq2seq(
            exp,
            seq2seq_config,
        )
        seq2seq_model = seq2seq_results["seq2seq_model"]
        seq2seq_results = seq2seq_results["seq2seq_results"]
    else:
        # Load the generated representations
        LOGGER.info(f"Loading final representations from numpy archive.")
        archive = np.load(npz, allow_pickle=True)
        seq2seq_results = {
            "targets": archive["targets"],
            "encodings": archive["encodings"],
            "embeddings": archive["embeddings"],
            "decodings": archive["decodings"],
            "decodings_embeddings": archive["decodings_embeddings"],
            "pw_seq_embeddings": archive["pw_seq_embeddings"].item(),
        }
    defrag(seq2seq_results, exp, defrag_config)
    gc.collect()


def run_all_permutations():
    adhoc_experiment = is_adhoc_experiment()
    if adhoc_experiment:
        raise ValueError()
        data_config_permutations = CATSYN_FEEDSTOCK.permute_configs()
        seq2seq_config = Seq2SeqConfigFeedstock().to_config()
    else:
        # The experiment is argv[1]
        experiment_name, experiment_configs = get_experiment()
        data_config_permutations, seq2seq_config, defrag_config = experiment_configs()
        seq2seq_config = seq2seq_config.to_config()
        # if isinstance(data_config_permutations, CatsynConfigFe edstock):
        data_config_permutations = data_config_permutations.permute_configs()

    for config_permutation in data_config_permutations:
        # For mimic experiments, we need to manually assign the names of the MIMIC variables we want to use
        if "cols" in (config_dict := config_permutation.to_config()):
            # seq2seq_config = Seq2SeqConfigFeedstock(COLS=config_dict["cols"]).to_config()
            seq2seq_config.cols = config_permutation.to_config()["cols"]
        run_permutation(
            config_permutation,
            seq2seq_config,
            defrag_config
        )
        # try:
        #     run_permutation(
        #         config_permutation,
        #         # cat2vec_config,
        #         seq2seq_config,
        #     )
        # except Exception as e:
        #     if CONTINUE_ON_ERROR:
        #         LOGGER.error(f"Experiment failed. Reason: {str(e)}")
        #         LOGGER.warning(
        #             f"Continuing onto next experiment because {CONTINUE_ON_ERROR=}."
        #         )
        #     else:
        #         raise RuntimeError() from e


if __name__ == "__main__":
    init_device()
    # clean_results()
    run_all_permutations()
    notify(f"Finished running {sys.argv[1]}.")
