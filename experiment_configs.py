# Config definitions for DEFRAG experiments
import dataclasses

from catsyn_config_generator import CatsynConfigFeedstock
from mimic_config_generator import MimicConfigFeedstock
from case_config_generator import Seq2SeqConfigFeedstock
from defrag import DefragConfig


# Reusable config components
# Catsyn
TWELVE_EXPERIMENT_CATSYN_CONFIG = CatsynConfigFeedstock(
    # ** DO NOT CHANGE **
    NB_PATIENTS=[1000],
    NB_STATES=[
        3,
        6,
        9,
        12,
    ],
    NB_VARIABLES=[3],
    DISTRIBUTION=[
        ("zipf_shuffle", {"a": 2}),
        ("zipf_shuffle", {"a": 3}),
        ("zipf_shuffle", {"a": 4}),
    ],
    MODEL=[
        ("directed_extended_barabasi_albert_graph", None),
    ],
    PERSISTENCE_MAX=[0.6],
    PERSISTENCE_MIN=[0.6],
)
# MIMIC
MIMIC_PROC_ONLY = MimicConfigFeedstock(
    # ** DO NOT CHANGE **
    CANCER_CCS_CODE=("2.5",),
    MIN_SEQ_LEN=8,
    TFIDF_DOC_COL="patient_id",
    TFIDF_VOCAB_COL="p_icd9_code",
    DIAGNOSIS_SELECTION='full',
    COLS=(
        "hadm_idx", 
        # "admission_type", "admission_location",  
        #"d_icd9_code", "d_ccs_lv1", "d_ccs_lv2", "d_ccs_lv3",
        "p_icd9_code", "p_ccs_lv1", "p_ccs_lv2", "p_ccs_lv3",
    ),
    DIAG_ONLY=False,
    PROC_ONLY=True,
)
# CaSE
CASE_50K_FOR_MIMIC = Seq2SeqConfigFeedstock(
    # ** DO NOT CHANGE **
    MODEL_TYPE="Seq2Seq",
    COLS=("VAR_0", "VAR_1", "VAR_2"),
    # This is the sequence length used during training
    L=8,
    # d_model: This must be the same as Cat2VecConfigFeedstock.N_STATE_FEATURES
    M=128,
    # num_heads: M must be divisible by N (d_model % num_heads)
    N=16,
    NUM_ENCODER_LAYERS=8,
    NUM_DECODER_LAYERS=8,
    DIM_FEEDFORWARD=64,
    DROPOUT=0.2,
    BATCH_SIZE=32,
    TRAIN_STEPS=50_000,
    VALIDATION_STEPS=25,
    P_MASK_THRESH=0.0,
    ENCODER_WINDOW_SIZE=1,  # this isn't actually used
)
# Defrag
DEFRAG_1K = DefragConfig(
    # ** DO NOT CHANGE **
    cluster_method="hdbscan",
    cluster_method_kwargs={"min_cluster_size": list(range(100, 800, 50)), "min_samples": [1]},
    cluster_metric="calinski_harabasz",
    exclude=False,
    infer=True,
    nproc=6,
)


"""
Synthetic Data Experiments 
Note: Each of the following two experiments needs to be run twice with slightly different
configurations to generate the full results used in the paper. The two parameters to change are:
CATSYN_FEEDSTOCK.NB_VARIABLES and SEQ2SEQ_CONFIG_FEEDSTOCK.

They need to be run once with:
    CATSYN_FEEDSTOCK.NB_VARIABLES = [1]
    SEQ2SEQ_CONFIG_FEEDSTOCK = ["VAR_0"]
And again with:
    CATSYN_FEEDSTOCK.NB_VARIABLES = [2]
    SEQ2SEQ_CONFIG_FEEDSTOCK = ["VAR_0", "VAR_1"]
"""

# Synthetic Data Experiments 
def synthetic_data_thesis_stlo_rpe_only_sep_hdbscan():  # Same for all clo sep con
    CATSYN_FEEDSTOCK = TWELVE_EXPERIMENT_CATSYN_CONFIG
    CATSYN_FEEDSTOCK.NB_STATES = [
        3, 5, 7, 9
    ]
    CATSYN_FEEDSTOCK.NB_VARIABLES = [1]
    CATSYN_FEEDSTOCK.NB_BINS = [100, 1000]
    CATSYN_FEEDSTOCK.DISTRIBUTION = [
        ("zipf_shuffle", {"a": 1.5}),
        ("zipf_shuffle", {"a": 2}),
        ("zipf_shuffle", {"a": 3}),
        # ("zipf_shuffle", {"a": 4}),
    ]
    CATSYN_FEEDSTOCK.SEED = [
        # 1, 
        2, 
        # 3,
    ]
    SEQ2SEQ_CONFIG_FEEDSTOCK = Seq2SeqConfigFeedstock(
        MODEL_TYPE="Seq2Seq",
        COLS=tuple([f"VAR_{i}" for i in range(CATSYN_FEEDSTOCK.NB_VARIABLES[0])]),
        # This is the sequence length used during training
        L=8,
        # d_model: This must be the same as Cat2VecConfigFeedstock.N_STATE_FEATURES
        M=64,
        # num_heads: M must be divisible by N (d_model % num_heads)
        N=16,
        NUM_ENCODER_LAYERS=4,
        NUM_DECODER_LAYERS=4,
        DIM_FEEDFORWARD=64,
        DROPOUT=0.2,
        BATCH_SIZE=64,
        TRAIN_STEPS=30_000,
        VALIDATION_STEPS=25,
        P_MASK_THRESH=0.0,
    )
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="hdbscan",
        cluster_method_kwargs={
            "min_cluster_size": list(range(5, 50, 5)) + list(range(50, 800, 50)), 
            "min_samples": [2, 5, 10, 50, 100]
        },
        cluster_metric="relative_validity",
        exclude=False,
        infer=True,
        nproc=1,
    )
    return CATSYN_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG

def synthetic_data_thesis_stlo_rpe_hierarchical():
    CATSYN_FEEDSTOCK = TWELVE_EXPERIMENT_CATSYN_CONFIG
    CATSYN_FEEDSTOCK.NB_STATES = [
        3, 5, 7, 9
    ]
    CATSYN_FEEDSTOCK.NB_VARIABLES = [2]
    # CATSYN_FEEDSTOCK.NB_VARIABLES = [2]
    # CATSYN_FEEDSTOCK.NB_BINS = [100]
    CATSYN_FEEDSTOCK.NB_BINS = [100]
    CATSYN_FEEDSTOCK.DISTRIBUTION = [
        ("zipf_shuffle", {"a": 1.5}),
        ("zipf_shuffle", {"a": 2}),
        ("zipf_shuffle", {"a": 3}),
        ("zipf_shuffle", {"a": 4}),
    ]
    CATSYN_FEEDSTOCK.SEED = [
        1, 
        2, 
        3,
    ]
    SEQ2SEQ_CONFIG_FEEDSTOCK = Seq2SeqConfigFeedstock(
        MODEL_TYPE="Seq2Seq",
        COLS=tuple([f"VAR_{i}" for i in range(CATSYN_FEEDSTOCK.NB_VARIABLES[0])]),
        # This is the sequence length used during training
        L=8,
        # d_model: This must be the same as Cat2VecConfigFeedstock.N_STATE_FEATURES
        M=64,
        # num_heads: M must be divisible by N (d_model % num_heads)
        N=16,
        NUM_ENCODER_LAYERS=4,
        NUM_DECODER_LAYERS=4,
        DIM_FEEDFORWARD=64,
        DROPOUT=0.2,
        BATCH_SIZE=64,
        TRAIN_STEPS=30_000,
        VALIDATION_STEPS=25,
        P_MASK_THRESH=0.0,
    )
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="hierarchical",
        cluster_method_kwargs={"n_clusters": [3, 5, 7, 9]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=6,
    )
    return CATSYN_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def synthetic_data_thesis_stlo_rpe_kmeans():
    CATSYN_FEEDSTOCK = TWELVE_EXPERIMENT_CATSYN_CONFIG
    CATSYN_FEEDSTOCK.NB_STATES = [
        3, 5, 7, 9
    ]
    CATSYN_FEEDSTOCK.NB_VARIABLES = [1]
    # CATSYN_FEEDSTOCK.NB_VARIABLES = [2]
    # CATSYN_FEEDSTOCK.NB_BINS = [100]
    CATSYN_FEEDSTOCK.NB_BINS = [100, 1000]
    CATSYN_FEEDSTOCK.DISTRIBUTION = [
        ("zipf_shuffle", {"a": 1.5}),
        ("zipf_shuffle", {"a": 2}),
        ("zipf_shuffle", {"a": 3}),
        ("zipf_shuffle", {"a": 4}),
    ]
    CATSYN_FEEDSTOCK.SEED = [
        1, 
        2, 
        3,
    ]
    SEQ2SEQ_CONFIG_FEEDSTOCK = Seq2SeqConfigFeedstock(
        MODEL_TYPE="Seq2Seq",
        COLS=tuple([f"VAR_{i}" for i in range(CATSYN_FEEDSTOCK.NB_VARIABLES[0])]),
        # This is the sequence length used during training
        L=8,
        # d_model: This must be the same as Cat2VecConfigFeedstock.N_STATE_FEATURES
        M=64,
        # num_heads: M must be divisible by N (d_model % num_heads)
        N=16,
        NUM_ENCODER_LAYERS=4,
        NUM_DECODER_LAYERS=4,
        DIM_FEEDFORWARD=64,
        DROPOUT=0.2,
        BATCH_SIZE=64,
        TRAIN_STEPS=30_000,
        VALIDATION_STEPS=25,
        P_MASK_THRESH=0.0,
    )
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="kmeans",
        cluster_method_kwargs={"n_clusters": [3, 5, 7, 9]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=1,
    )
    return CATSYN_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def synthetic_data_thesis_mse_rpe_hierarchical():
    CATSYN_FEEDSTOCK = TWELVE_EXPERIMENT_CATSYN_CONFIG
    CATSYN_FEEDSTOCK.NB_STATES = [
        3, 5, 7, 9
    ]
    CATSYN_FEEDSTOCK.NB_VARIABLES = [1]
    # CATSYN_FEEDSTOCK.NB_VARIABLES = [2]
    # CATSYN_FEEDSTOCK.NB_BINS = [100]
    CATSYN_FEEDSTOCK.NB_BINS = [1000]
    CATSYN_FEEDSTOCK.DISTRIBUTION = [
        ("zipf_shuffle", {"a": 1.5}),
        ("zipf_shuffle", {"a": 2}),
        ("zipf_shuffle", {"a": 3}),
        ("zipf_shuffle", {"a": 4}),
    ]
    CATSYN_FEEDSTOCK.SEED = [
        1, 
        2, 
        3,
    ]
    SEQ2SEQ_CONFIG_FEEDSTOCK = Seq2SeqConfigFeedstock(
        MODEL_TYPE="Seq2Seq",
        COLS=tuple([f"VAR_{i}" for i in range(CATSYN_FEEDSTOCK.NB_VARIABLES[0])]),
        # This is the sequence length used during training
        L=8,
        # d_model: This must be the same as Cat2VecConfigFeedstock.N_STATE_FEATURES
        M=64,
        # num_heads: M must be divisible by N (d_model % num_heads)
        N=16,
        NUM_ENCODER_LAYERS=4,
        NUM_DECODER_LAYERS=4,
        DIM_FEEDFORWARD=64,
        DROPOUT=0.2,
        BATCH_SIZE=64,
        TRAIN_STEPS=30_000,
        VALIDATION_STEPS=25,
        P_MASK_THRESH=0.0,
    )
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="hierarchical",
        cluster_method_kwargs={"n_clusters": [3, 5, 7, 9]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=1,
    )
    return CATSYN_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def synthetic_data_thesis_stlo_npe_hdbscan():
    CATSYN_FEEDSTOCK = TWELVE_EXPERIMENT_CATSYN_CONFIG
    CATSYN_FEEDSTOCK.NB_STATES = [
        3, 5, 7, 9
    ]
    CATSYN_FEEDSTOCK.NB_VARIABLES = [1]
    CATSYN_FEEDSTOCK.NB_BINS = [100, 1000]
    CATSYN_FEEDSTOCK.DISTRIBUTION = [
        ("zipf_shuffle", {"a": 1.5}),
        ("zipf_shuffle", {"a": 2}),
        ("zipf_shuffle", {"a": 3}),
        ("zipf_shuffle", {"a": 4}),
    ]
    CATSYN_FEEDSTOCK.SEED = [
        1, 
        2, 
        3,
    ]
    SEQ2SEQ_CONFIG_FEEDSTOCK = Seq2SeqConfigFeedstock(
        MODEL_TYPE="Seq2Seq",
        COLS=tuple([f"VAR_{i}" for i in range(CATSYN_FEEDSTOCK.NB_VARIABLES[0])]),
        # This is the sequence length used during training
        L=8,
        # d_model: This must be the same as Cat2VecConfigFeedstock.N_STATE_FEATURES
        M=64,
        # num_heads: M must be divisible by N (d_model % num_heads)
        N=16,
        NUM_ENCODER_LAYERS=4,
        NUM_DECODER_LAYERS=4,
        DIM_FEEDFORWARD=64,
        DROPOUT=0.2,
        BATCH_SIZE=64,
        TRAIN_STEPS=30_000,
        VALIDATION_STEPS=25,
        P_MASK_THRESH=0.0,
    )
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="hdbscan",
        cluster_method_kwargs={
            "min_cluster_size": list(range(5, 50, 5)) + list(range(50, 800, 50)), 
            "min_samples": [2, 5, 10, 50, 100]
        },
        cluster_metric="relative_validity",
        exclude=False,
        infer=True,
        nproc=1,
    )
    return CATSYN_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def synthetic_data_thesis_stlo_ape_hdbscan():
    CATSYN_FEEDSTOCK = TWELVE_EXPERIMENT_CATSYN_CONFIG
    CATSYN_FEEDSTOCK.NB_STATES = [
        3, 5, 7, 9
    ]
    CATSYN_FEEDSTOCK.NB_VARIABLES = [2]
    CATSYN_FEEDSTOCK.NB_BINS = [100, 1000]
    CATSYN_FEEDSTOCK.DISTRIBUTION = [
        ("zipf_shuffle", {"a": 1.5}),
        ("zipf_shuffle", {"a": 2}),
        ("zipf_shuffle", {"a": 3}),
        ("zipf_shuffle", {"a": 4}),
    ]
    CATSYN_FEEDSTOCK.SEED = [
        1, 
        2, 
        3,
    ]
    SEQ2SEQ_CONFIG_FEEDSTOCK = Seq2SeqConfigFeedstock(
        MODEL_TYPE="Seq2Seq",
        COLS=tuple([f"VAR_{i}" for i in range(CATSYN_FEEDSTOCK.NB_VARIABLES[0])]),
        # This is the sequence length used during training
        L=8,
        # d_model: This must be the same as Cat2VecConfigFeedstock.N_STATE_FEATURES
        M=64,
        # num_heads: M must be divisible by N (d_model % num_heads)
        N=16,
        NUM_ENCODER_LAYERS=4,
        NUM_DECODER_LAYERS=4,
        DIM_FEEDFORWARD=64,
        DROPOUT=0.2,
        BATCH_SIZE=64,
        TRAIN_STEPS=30_000,
        VALIDATION_STEPS=25,
        P_MASK_THRESH=0.0,
    )
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="hdbscan",
        cluster_method_kwargs={
            "min_cluster_size": list(range(5, 50, 5)) + list(range(50, 800, 50)), 
            "min_samples": [2, 5, 10, 50, 100]
        },
        cluster_metric="relative_validity",
        exclude=False,
        infer=True,
        nproc=1,
    )
    return CATSYN_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def synthetic_data_thesis_stlo_rpe_hdbscan():
    CATSYN_FEEDSTOCK = TWELVE_EXPERIMENT_CATSYN_CONFIG
    CATSYN_FEEDSTOCK.NB_STATES = [
        3, 5, 7, 9
    ]
    CATSYN_FEEDSTOCK.NB_VARIABLES = [1]
    CATSYN_FEEDSTOCK.NB_BINS = [100, 1000]
    CATSYN_FEEDSTOCK.DISTRIBUTION = [
        ("zipf_shuffle", {"a": 1.5}),
        ("zipf_shuffle", {"a": 2}),
        ("zipf_shuffle", {"a": 3}),
        ("zipf_shuffle", {"a": 4}),
    ]
    CATSYN_FEEDSTOCK.SEED = [
        1, 
        2, 
        3,
    ]
    SEQ2SEQ_CONFIG_FEEDSTOCK = Seq2SeqConfigFeedstock(
        MODEL_TYPE="Seq2Seq",
        COLS=tuple([f"VAR_{i}" for i in range(CATSYN_FEEDSTOCK.NB_VARIABLES[0])]),
        # This is the sequence length used during training
        L=8,
        # d_model: This must be the same as Cat2VecConfigFeedstock.N_STATE_FEATURES
        M=64,
        # num_heads: M must be divisible by N (d_model % num_heads)
        N=16,
        NUM_ENCODER_LAYERS=4,
        NUM_DECODER_LAYERS=4,
        DIM_FEEDFORWARD=64,
        DROPOUT=0.2,
        BATCH_SIZE=64,
        TRAIN_STEPS=30_000,
        VALIDATION_STEPS=25,
        P_MASK_THRESH=0.0,
    )
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="hdbscan",
        cluster_method_kwargs={
            "min_cluster_size": list(range(5, 50, 5)) + list(range(50, 800, 50)), 
            "min_samples": [2, 5, 10, 50, 100]
        },
        cluster_metric="relative_validity",
        exclude=False,
        infer=True,
        nproc=1,
    )
    return CATSYN_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def synthetic_data_thesis_mse_rpe_hdbscan():
    CATSYN_FEEDSTOCK = TWELVE_EXPERIMENT_CATSYN_CONFIG
    CATSYN_FEEDSTOCK.NB_STATES = [
        3, 5, 7, 9
    ]
    CATSYN_FEEDSTOCK.NB_VARIABLES = [1]
    CATSYN_FEEDSTOCK.NB_BINS = [100, 1000]
    CATSYN_FEEDSTOCK.DISTRIBUTION = [
        ("zipf_shuffle", {"a": 1.5}),
        ("zipf_shuffle", {"a": 2}),
        ("zipf_shuffle", {"a": 3}),
        ("zipf_shuffle", {"a": 4}),
    ]
    CATSYN_FEEDSTOCK.SEED = [
        1, 
        2, 
        3,
    ]
    SEQ2SEQ_CONFIG_FEEDSTOCK = Seq2SeqConfigFeedstock(
        MODEL_TYPE="Seq2Seq",
        COLS=tuple([f"VAR_{i}" for i in range(CATSYN_FEEDSTOCK.NB_VARIABLES[0])]),
        # This is the sequence length used during training
        L=8,
        # d_model: This must be the same as Cat2VecConfigFeedstock.N_STATE_FEATURES
        M=64,
        # num_heads: M must be divisible by N (d_model % num_heads)
        N=16,
        NUM_ENCODER_LAYERS=4,
        NUM_DECODER_LAYERS=4,
        DIM_FEEDFORWARD=64,
        DROPOUT=0.2,
        BATCH_SIZE=64,
        TRAIN_STEPS=30_000,
        VALIDATION_STEPS=25,
        P_MASK_THRESH=0.0,
    )
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="hdbscan",
        cluster_method_kwargs={
            "min_cluster_size": list(range(5, 50, 5)) + list(range(50, 800, 50)), 
            "min_samples": [2, 5, 10, 50, 100]
        },
        cluster_metric="relative_validity",
        exclude=False,
        infer=True,
        nproc=0,
    )
    return CATSYN_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def synthetic_data_thesis_barlow_rpe_hdbscan():
    CATSYN_FEEDSTOCK = TWELVE_EXPERIMENT_CATSYN_CONFIG
    CATSYN_FEEDSTOCK.NB_STATES = [
        3, 5, 7, 9
    ]
    CATSYN_FEEDSTOCK.NB_VARIABLES = [1]
    CATSYN_FEEDSTOCK.NB_BINS = [100, 1000]
    CATSYN_FEEDSTOCK.DISTRIBUTION = [
        ("zipf_shuffle", {"a": 1.5}),
        ("zipf_shuffle", {"a": 2}),
        ("zipf_shuffle", {"a": 3}),
        ("zipf_shuffle", {"a": 4}),
    ]
    CATSYN_FEEDSTOCK.SEED = [
        1, 
        2, 
        3,
    ]
    SEQ2SEQ_CONFIG_FEEDSTOCK = Seq2SeqConfigFeedstock(
        MODEL_TYPE="Seq2Seq",
        COLS=tuple([f"VAR_{i}" for i in range(CATSYN_FEEDSTOCK.NB_VARIABLES[0])]),
        # This is the sequence length used during training
        L=8,
        # d_model: This must be the same as Cat2VecConfigFeedstock.N_STATE_FEATURES
        M=64,
        # num_heads: M must be divisible by N (d_model % num_heads)
        N=16,
        NUM_ENCODER_LAYERS=4,
        NUM_DECODER_LAYERS=4,
        DIM_FEEDFORWARD=64,
        DROPOUT=0.2,
        BATCH_SIZE=64,
        TRAIN_STEPS=30_000,
        VALIDATION_STEPS=25,
        P_MASK_THRESH=0.0,
    )
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="hdbscan",
        cluster_method_kwargs={
            "min_cluster_size": list(range(5, 50, 5)) + list(range(50, 800, 50)), 
            "min_samples": [2, 5, 10, 50, 100]
        },
        cluster_metric="relative_validity",
        exclude=False,
        infer=True,
        nproc=1,
    )
    return CATSYN_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def synthetic_data_thesis_simcse_rpe_hdbscan():
    CATSYN_FEEDSTOCK = TWELVE_EXPERIMENT_CATSYN_CONFIG
    CATSYN_FEEDSTOCK.NB_STATES = [
        3, 5, 7, 9
    ]
    CATSYN_FEEDSTOCK.NB_VARIABLES = [1]
    CATSYN_FEEDSTOCK.NB_BINS = [100, 1000]
    CATSYN_FEEDSTOCK.DISTRIBUTION = [
        ("zipf_shuffle", {"a": 1.5}),
        ("zipf_shuffle", {"a": 2}),
        ("zipf_shuffle", {"a": 3}),
        ("zipf_shuffle", {"a": 4}),
    ]
    CATSYN_FEEDSTOCK.SEED = [
        1, 
        2, 
        3,
    ]
    SEQ2SEQ_CONFIG_FEEDSTOCK = Seq2SeqConfigFeedstock(
        MODEL_TYPE="Seq2Seq",
        COLS=tuple([f"VAR_{i}" for i in range(CATSYN_FEEDSTOCK.NB_VARIABLES[0])]),
        # This is the sequence length used during training
        L=8,
        # d_model: This must be the same as Cat2VecConfigFeedstock.N_STATE_FEATURES
        M=64,
        # num_heads: M must be divisible by N (d_model % num_heads)
        N=16,
        NUM_ENCODER_LAYERS=4,
        NUM_DECODER_LAYERS=4,
        DIM_FEEDFORWARD=64,
        DROPOUT=0.2,
        BATCH_SIZE=64,
        TRAIN_STEPS=30_000,
        VALIDATION_STEPS=25,
        P_MASK_THRESH=0.0,
    )
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="hdbscan",
        cluster_method_kwargs={
            "min_cluster_size": list(range(5, 50, 5)) + list(range(50, 800, 50)), 
            "min_samples": [2, 5, 10, 50, 100]
        },
        cluster_metric="relative_validity",
        exclude=False,
        infer=True,
        nproc=1,
    )
    return CATSYN_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG

def synthetic_data_loss_on_encoder_mse():
    CATSYN_FEEDSTOCK = TWELVE_EXPERIMENT_CATSYN_CONFIG
    CATSYN_FEEDSTOCK.NB_STATES = [3]
    CATSYN_FEEDSTOCK.NB_VARIABLES = [1]
    CATSYN_FEEDSTOCK.DISTRIBUTION = [
        ("zipf_shuffle", {"a": 2}),
    ]
    CATSYN_FEEDSTOCK.SEED = [
        1, 
    ]
    SEQ2SEQ_CONFIG_FEEDSTOCK = Seq2SeqConfigFeedstock(
        MODEL_TYPE="Seq2Seq",
        COLS=("VAR_0",),
        # This is the sequence length used during training
        L=8,
        # d_model: This must be the same as Cat2VecConfigFeedstock.N_STATE_FEATURES
        M=64,
        # num_heads: M must be divisible by N (d_model % num_heads)
        N=16,
        NUM_ENCODER_LAYERS=4,
        NUM_DECODER_LAYERS=4,
        DIM_FEEDFORWARD=64,
        DROPOUT=0.2,
        BATCH_SIZE=64,
        TRAIN_STEPS=30_000,
        VALIDATION_STEPS=25,
        P_MASK_THRESH=0.0,
    )
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="hierarchical",
        cluster_method_kwargs={"n_clusters": [3, 5, 7, 9]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=6,
    )
    return CATSYN_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
# The big Synthetic Data experiment
def synthetic_data_experiment_big():
    CATSYN_FEEDSTOCK = TWELVE_EXPERIMENT_CATSYN_CONFIG
    CATSYN_FEEDSTOCK.NB_STATES = [3, 6, 9, 9]
    CATSYN_FEEDSTOCK.NB_VARIABLES = [2]
    CATSYN_FEEDSTOCK.DISTRIBUTION = [
        ("zipf_shuffle", {"a": 1.5}),
        ("zipf_shuffle", {"a": 2}),
        ("zipf_shuffle", {"a": 3}),
        ("zipf_shuffle", {"a": 4}),
    ]
    CATSYN_FEEDSTOCK.SEED = [
        1, 
        2, 
        3,
    ]
    SEQ2SEQ_CONFIG_FEEDSTOCK = Seq2SeqConfigFeedstock(
        MODEL_TYPE="Seq2Seq",
        COLS=("VAR_0","VAR_1"),
        # This is the sequence length used during training
        L=8,
        # d_model: This must be the same as Cat2VecConfigFeedstock.N_STATE_FEATURES
        M=64,
        # num_heads: M must be divisible by N (d_model % num_heads)
        N=16,
        NUM_ENCODER_LAYERS=4,
        NUM_DECODER_LAYERS=4,
        DIM_FEEDFORWARD=64,
        DROPOUT=0.2,
        BATCH_SIZE=64,
        TRAIN_STEPS=30_000,
        VALIDATION_STEPS=25,
        P_MASK_THRESH=0.0,
    )
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="hdbscan",
        cluster_method_kwargs={
            "min_cluster_size": list(range(5, 50, 5)) + list(range(50, 800, 50)), 
            "min_samples": [2, 5, 10, 50, 100]
        },
        cluster_metric="relative_validity",
        exclude=False,
        infer=True,
        nproc=6,
    )
    return CATSYN_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def synthetic_data_experiment_big_1000_bins():
    CATSYN_FEEDSTOCK = TWELVE_EXPERIMENT_CATSYN_CONFIG
    CATSYN_FEEDSTOCK.NB_BINS = [1000]
    CATSYN_FEEDSTOCK.NB_STATES = [3, 5, 7, 9]
    CATSYN_FEEDSTOCK.NB_VARIABLES = [1]
    CATSYN_FEEDSTOCK.DISTRIBUTION = [
        ("zipf_shuffle", {"a": 1.5}),
        ("zipf_shuffle", {"a": 2}),
        ("zipf_shuffle", {"a": 3}),
        ("zipf_shuffle", {"a": 4}),
    ]
    CATSYN_FEEDSTOCK.SEED = [
        1, 
        2, 
        3,
    ]
    SEQ2SEQ_CONFIG_FEEDSTOCK = Seq2SeqConfigFeedstock(
        MODEL_TYPE="Seq2Seq",
        COLS=("VAR_0",),
        # This is the sequence length used during training
        L=8,
        # d_model: This must be the same as Cat2VecConfigFeedstock.N_STATE_FEATURES
        M=64,
        # num_heads: M must be divisible by N (d_model % num_heads)
        N=16,
        NUM_ENCODER_LAYERS=4,
        NUM_DECODER_LAYERS=4,
        DIM_FEEDFORWARD=64,
        DROPOUT=0.2,
        BATCH_SIZE=64,
        TRAIN_STEPS=30_000,
        VALIDATION_STEPS=25,
        P_MASK_THRESH=0.0,
    )
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="hdbscan",
        cluster_method_kwargs={
            "min_cluster_size": list(range(5, 50, 5)) + list(range(50, 800, 50)), 
            "min_samples": [2, 5, 10, 50, 100]
        },
        cluster_metric="relative_validity",
        exclude=False,
        infer=True,
        nproc=6,
    )
    return CATSYN_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG

def synthetic_data_experiment_big_hierarchical():
    CATSYN_FEEDSTOCK = TWELVE_EXPERIMENT_CATSYN_CONFIG
    CATSYN_FEEDSTOCK.NB_STATES = [3, 5, 7, 9]
    CATSYN_FEEDSTOCK.NB_VARIABLES = [1]
    CATSYN_FEEDSTOCK.DISTRIBUTION = [
        ("zipf_shuffle", {"a": 1.5}),
        ("zipf_shuffle", {"a": 2}),
        ("zipf_shuffle", {"a": 3}),
        ("zipf_shuffle", {"a": 4}),
    ]
    CATSYN_FEEDSTOCK.SEED = [
        1, 
        2, 
        3,
    ]
    SEQ2SEQ_CONFIG_FEEDSTOCK = Seq2SeqConfigFeedstock(
        MODEL_TYPE="Seq2Seq",
        COLS=tuple([f"VAR_{i}" for i in range(CATSYN_FEEDSTOCK.NB_VARIABLES)]),
        # This is the sequence length used during training
        L=8,
        # d_model: This must be the same as Cat2VecConfigFeedstock.N_STATE_FEATURES
        M=64,
        # num_heads: M must be divisible by N (d_model % num_heads)
        N=16,
        NUM_ENCODER_LAYERS=4,
        NUM_DECODER_LAYERS=4,
        DIM_FEEDFORWARD=64,
        DROPOUT=0.2,
        BATCH_SIZE=64,
        TRAIN_STEPS=30_000,
        VALIDATION_STEPS=25,
        P_MASK_THRESH=0.0,
    )
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="hierarchical",
        cluster_method_kwargs={"n_clusters": [3, 5, 7, 9]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=6,
    )
    return CATSYN_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def synthetic_data_experiment_big_1000_bins_hierarchical():
    CATSYN_FEEDSTOCK = TWELVE_EXPERIMENT_CATSYN_CONFIG
    CATSYN_FEEDSTOCK.NB_BINS = [1000]
    CATSYN_FEEDSTOCK.NB_STATES = [3, 5, 7, 9]
    CATSYN_FEEDSTOCK.NB_VARIABLES = [2]
    CATSYN_FEEDSTOCK.DISTRIBUTION = [
        ("zipf_shuffle", {"a": 1.5}),
        ("zipf_shuffle", {"a": 2}),
        ("zipf_shuffle", {"a": 3}),
        ("zipf_shuffle", {"a": 4}),
    ]
    CATSYN_FEEDSTOCK.SEED = [
        1, 
        2, 
        3,
    ]
    SEQ2SEQ_CONFIG_FEEDSTOCK = Seq2SeqConfigFeedstock(
        MODEL_TYPE="Seq2Seq",
        COLS=("VAR_0","VAR_1",),
        # This is the sequence length used during training
        L=8,
        # d_model: This must be the same as Cat2VecConfigFeedstock.N_STATE_FEATURES
        M=64,
        # num_heads: M must be divisible by N (d_model % num_heads)
        N=16,
        NUM_ENCODER_LAYERS=4,
        NUM_DECODER_LAYERS=4,
        DIM_FEEDFORWARD=64,
        DROPOUT=0.2,
        BATCH_SIZE=64,
        TRAIN_STEPS=30_000,
        VALIDATION_STEPS=25,
        P_MASK_THRESH=0.0,
    )
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="hierarchical",
        cluster_method_kwargs={"n_clusters": [3, 5, 7, 9]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=6,
    )
    return CATSYN_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG

"""
MIMIC-IV Experiments
"""
def mimic_experiment_breast_soft_hierarchical_5():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.MIN_SEQ_LEN = 5
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("2.5",)
    MIMIC_FEEDSTOCK.DIAGNOSIS_SELECTION = "soft"
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    SEQ2SEQ_CONFIG_FEEDSTOCK.ENCODER_WINDOW_SIZE = 5
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="hierarchical",
        cluster_method_kwargs={"n_clusters": [5]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=6,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def mimic_experiment_breast_soft_hierarchical_5_tril():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.MIN_SEQ_LEN = 5
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("2.5",)
    MIMIC_FEEDSTOCK.DIAGNOSIS_SELECTION = "soft"
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    SEQ2SEQ_CONFIG_FEEDSTOCK.ENCODER_WINDOW_SIZE = 5
    # DEFRAG_CONFIG = DefragConfig(
    #     cluster_method="kmeans",
    #     cluster_method_kwargs={"n_clusters": [20]},
    #     cluster_metric="calinski_harabasz",
    #     exclude=False,
    #     infer=True,
    #     nproc=6,
    #     colour_by="cluster",
    # )
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="hierarchical",
        cluster_method_kwargs={"n_clusters": [5]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=6,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def mimic_experiment_lung_soft_hierarchical_5():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.MIN_SEQ_LEN = 6
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("2.3",)
    MIMIC_FEEDSTOCK.DIAGNOSIS_SELECTION = "soft"
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    SEQ2SEQ_CONFIG_FEEDSTOCK.ENCODER_WINDOW_SIZE = 2
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="hierarchical",
        cluster_method_kwargs={"n_clusters": [5]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=6,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def mimic_experiment_melanoma_soft_hierarchical_5():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.MIN_SEQ_LEN = 5
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("2.4.1",)
    MIMIC_FEEDSTOCK.DIAGNOSIS_SELECTION = "soft"
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    SEQ2SEQ_CONFIG_FEEDSTOCK.ENCODER_WINDOW_SIZE = 5
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="hierarchical",
        cluster_method_kwargs={"n_clusters": [5]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=6,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG

def mimic_experiment_colon_soft_hierarchical_5():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("2.1",)
    MIMIC_FEEDSTOCK.DIAGNOSIS_SELECTION = "soft"
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="hierarchical",
        cluster_method_kwargs={"n_clusters": [5]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=6,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG

# (
#   MIMIC_FEEDSTOCK.MIN_SEQ_LEN - 
#   SEQ2SEQ_CONFIG_FEEDSTOCK.L +
#   SEQ2SEQ_CONFIG_FEEDSTOCK.ENCODER_WINDOW_SIZE
# ) >= 0
def mimic_experiment_breast_soft_window_0():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.MIN_SEQ_LEN = 8
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("2.5",)
    MIMIC_FEEDSTOCK.DIAGNOSIS_SELECTION = "soft"
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    SEQ2SEQ_CONFIG_FEEDSTOCK.ENCODER_WINDOW_SIZE = 0
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="hierarchical",
        cluster_method_kwargs={"n_clusters": [5]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=6,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def mimic_experiment_breast_soft_window_1():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.MIN_SEQ_LEN = 8
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("2.5",)
    MIMIC_FEEDSTOCK.DIAGNOSIS_SELECTION = "soft"
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    SEQ2SEQ_CONFIG_FEEDSTOCK.ENCODER_WINDOW_SIZE = 1
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="hierarchical",
        cluster_method_kwargs={"n_clusters": [5]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=6,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def mimic_experiment_breast_soft_window_2():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.MIN_SEQ_LEN = 8
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("2.5",)
    MIMIC_FEEDSTOCK.DIAGNOSIS_SELECTION = "soft"
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    SEQ2SEQ_CONFIG_FEEDSTOCK.ENCODER_WINDOW_SIZE = 2
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="hierarchical",
        cluster_method_kwargs={"n_clusters": [5]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=6,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def mimic_experiment_breast_soft_window_3():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.MIN_SEQ_LEN = 8
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("2.5",)
    MIMIC_FEEDSTOCK.DIAGNOSIS_SELECTION = "soft"
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    SEQ2SEQ_CONFIG_FEEDSTOCK.ENCODER_WINDOW_SIZE = 3
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="hierarchical",
        cluster_method_kwargs={"n_clusters": [5]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=6,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def mimic_experiment_breast_soft_window_5():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.MIN_SEQ_LEN = 8
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("2.5",)
    MIMIC_FEEDSTOCK.DIAGNOSIS_SELECTION = "soft"
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    SEQ2SEQ_CONFIG_FEEDSTOCK.ENCODER_WINDOW_SIZE = 5
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="hierarchical",
        cluster_method_kwargs={"n_clusters": [5]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=6,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def mimic_experiment_breast_soft_window_7():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.MIN_SEQ_LEN = 8
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("2.5",)
    MIMIC_FEEDSTOCK.DIAGNOSIS_SELECTION = "soft"
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    SEQ2SEQ_CONFIG_FEEDSTOCK.ENCODER_WINDOW_SIZE = 7
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="hierarchical",
        cluster_method_kwargs={"n_clusters": [5]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=6,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def mimic_experiment_breast_soft_window_10():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.MIN_SEQ_LEN = 8
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("2.5",)
    MIMIC_FEEDSTOCK.DIAGNOSIS_SELECTION = "soft"
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    SEQ2SEQ_CONFIG_FEEDSTOCK.ENCODER_WINDOW_SIZE = 10
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="hierarchical",
        cluster_method_kwargs={"n_clusters": [5]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=6,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def mimic_experiment_breast_soft_window_15():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.MIN_SEQ_LEN = 8
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("2.5",)
    MIMIC_FEEDSTOCK.DIAGNOSIS_SELECTION = "soft"
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    SEQ2SEQ_CONFIG_FEEDSTOCK.ENCODER_WINDOW_SIZE = 15
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="hierarchical",
        cluster_method_kwargs={"n_clusters": [5]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=6,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def mimic_experiment_breast_soft_window_20():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.MIN_SEQ_LEN = 8
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("2.5",)
    MIMIC_FEEDSTOCK.DIAGNOSIS_SELECTION = "soft"
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    SEQ2SEQ_CONFIG_FEEDSTOCK.ENCODER_WINDOW_SIZE = 20
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="hierarchical",
        cluster_method_kwargs={"n_clusters": [5]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=6,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG

#   SOFT Colon
def mimic_experiment_colon_soft_km2():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("2.1",)
    MIMIC_FEEDSTOCK.DIAGNOSIS_SELECTION = "soft"
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="kmeans",
        cluster_method_kwargs={"n_clusters": [2]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=6,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def mimic_experiment_colon_soft_km3():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("2.1",)
    MIMIC_FEEDSTOCK.DIAGNOSIS_SELECTION = "soft"
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="kmeans",
        cluster_method_kwargs={"n_clusters": [3]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=6,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def mimic_experiment_colon_soft_km4():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("2.1",)
    MIMIC_FEEDSTOCK.DIAGNOSIS_SELECTION = "soft"
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="kmeans",
        cluster_method_kwargs={"n_clusters": [4]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=6,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def mimic_experiment_colon_soft_km5():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("2.1",)
    MIMIC_FEEDSTOCK.DIAGNOSIS_SELECTION = "soft"
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="kmeans",
        cluster_method_kwargs={"n_clusters": [5]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=6,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def mimic_experiment_colon_soft_km6():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("2.1",)
    MIMIC_FEEDSTOCK.DIAGNOSIS_SELECTION = "soft"
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="kmeans",
        cluster_method_kwargs={"n_clusters": [6]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=6,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def mimic_experiment_colon_soft_km7():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("2.1",)
    MIMIC_FEEDSTOCK.DIAGNOSIS_SELECTION = "soft"
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="kmeans",
        cluster_method_kwargs={"n_clusters": [7]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=6,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def mimic_experiment_colon_soft_km8():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("2.1",)
    MIMIC_FEEDSTOCK.DIAGNOSIS_SELECTION = "soft"
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="kmeans",
        cluster_method_kwargs={"n_clusters": [8]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=6,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
#   HARD filtering experiments
#        Colon
def mimic_experiment_colon_hard_km2():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("2.1",)
    MIMIC_FEEDSTOCK.DIAGNOSIS_SELECTION = "hard"
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="kmeans",
        cluster_method_kwargs={"n_clusters": [2]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=6,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def mimic_experiment_colon_hard_km3():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("2.1",)
    MIMIC_FEEDSTOCK.DIAGNOSIS_SELECTION = "hard"
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="kmeans",
        cluster_method_kwargs={"n_clusters": [3]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=6,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def mimic_experiment_colon_hard_km4():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("2.1",)
    MIMIC_FEEDSTOCK.DIAGNOSIS_SELECTION = "hard"
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="kmeans",
        cluster_method_kwargs={"n_clusters": [4]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=6,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def mimic_experiment_colon_hard_km5():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("2.1",)
    MIMIC_FEEDSTOCK.DIAGNOSIS_SELECTION = "hard"
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="kmeans",
        cluster_method_kwargs={"n_clusters": [5]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=6,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def mimic_experiment_colon_hard_km6():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("2.1",)
    MIMIC_FEEDSTOCK.DIAGNOSIS_SELECTION = "hard"
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="kmeans",
        cluster_method_kwargs={"n_clusters": [6]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=6,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def mimic_experiment_colon_hard_km7():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("2.1",)
    MIMIC_FEEDSTOCK.DIAGNOSIS_SELECTION = "hard"
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="kmeans",
        cluster_method_kwargs={"n_clusters": [7]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=6,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def mimic_experiment_colon_hard_km8():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("2.1",)
    MIMIC_FEEDSTOCK.DIAGNOSIS_SELECTION = "hard"
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="kmeans",
        cluster_method_kwargs={"n_clusters": [8]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=6,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def mimic_experiment_colon_hard_hdbscan():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("2.1",)
    MIMIC_FEEDSTOCK.DIAGNOSIS_SELECTION = "hard"
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="hdbscan",
        cluster_method_kwargs={"min_cluster_size": list(range(10, 200, 10)), "min_samples": [1]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=6,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG

# Original Breast cancer experiments
def mimic_experiment_breast_intra():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("2.5",)
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    DEFRAG_CONFIG = DEFRAG_1K
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def mimic_experiment_breast_intra_lower_cluster_params():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("2.5",)
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    DEFRAG_CONFIG = DEFRAG_1K
    DEFRAG_CONFIG.min_cluster_size_range = list(range(10, 100, 10)) + list(range(100, 1000, 10))
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def mimic_experiment_breast_intra_100k():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("2.5",)
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    SEQ2SEQ_CONFIG_FEEDSTOCK.TRAIN_STEPS = 100_000
    DEFRAG_CONFIG = DEFRAG_1K
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def mimic_experiment_breast_intra_bigger_model():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("2.5",)
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    SEQ2SEQ_CONFIG_FEEDSTOCK.M = 256
    SEQ2SEQ_CONFIG_FEEDSTOCK.NUM_ENCODER_LAYERS = 16
    SEQ2SEQ_CONFIG_FEEDSTOCK.NUM_DECODER_LAYERS = 16
    SEQ2SEQ_CONFIG_FEEDSTOCK.DIM_FEEDFORWARD = 256
    DEFRAG_CONFIG = DEFRAG_1K
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
# Sweep
def mimic_experiment_breast_intra_kmeans_2():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("2.5",)
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="kmeans",
        cluster_method_kwargs={"n_clusters": [2]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=6,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def mimic_experiment_breast_intra_kmeans_3():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("2.5",)
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="kmeans",
        cluster_method_kwargs={"n_clusters": [3]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=6,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def mimic_experiment_breast_intra_kmeans_4():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("2.5",)
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="kmeans",
        cluster_method_kwargs={"n_clusters": [4]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=6,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def mimic_experiment_breast_intra_kmeans_5():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("2.5",)
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="kmeans",
        cluster_method_kwargs={"n_clusters": [5]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=6,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def mimic_experiment_breast_intra_kmeans_6():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("2.5",)
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="kmeans",
        cluster_method_kwargs={"n_clusters": [6]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=6,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def mimic_experiment_breast_intra_kmeans_7():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("2.5",)
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="kmeans",
        cluster_method_kwargs={"n_clusters": [7]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=6,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def mimic_experiment_breast_intra_kmeans_8():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("2.5",)
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="kmeans",
        cluster_method_kwargs={"n_clusters": [8]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=6,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def mimic_experiment_breast_intra_kmeans_9():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("2.5",)
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="kmeans",
        cluster_method_kwargs={"n_clusters": [9]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=6,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def mimic_experiment_breast_intra_kmeans_10():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("2.5",)
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="kmeans",
        cluster_method_kwargs={"n_clusters": [10]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=6,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
# Other
def mimic_experiment_breast_intra_kmeans_5_with_hadm_idx():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("2.5",)
    MIMIC_FEEDSTOCK.COLS=("hadm_idx", "p_icd9_code", "p_ccs_lv1", "p_ccs_lv2", "p_ccs_lv3")
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="kmeans",
        cluster_method_kwargs={"n_clusters": [5]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=6,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def mimic_experiment_appendicitis_kmeans_5():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("9.6.1",)
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="kmeans",
        cluster_method_kwargs={"n_clusters": [5]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=6,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def mimic_experiment_colorectal_kmeans_5():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("2.1",)
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="kmeans",
        cluster_method_kwargs={"n_clusters": [5]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=6,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def mimic_experiment_colorectal_kmeans_4():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("2.1",)
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="kmeans",
        cluster_method_kwargs={"n_clusters": [4]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=6,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def mimic_experiment_colorectal_kmeans_3():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("2.1",)
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="kmeans",
        cluster_method_kwargs={"n_clusters": [3]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=6,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def mimic_experiment_colorectal_kmeans_2():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("2.1",)
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="kmeans",
        cluster_method_kwargs={"n_clusters": [2]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=6,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def mimic_experiment_lung_cancer_kmeans_5():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("2.3",)
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="kmeans",
        cluster_method_kwargs={"n_clusters": [5]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=6,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def mimic_experiment_skin_cancer_kmeans_5():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("2.4",)
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="kmeans",
        cluster_method_kwargs={"n_clusters": [5]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=6,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
# Only primary diagnosis
def mimic_experiment_breast_primary_soft():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("2.5",)
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="kmeans",
        cluster_method_kwargs={"n_clusters": [5]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=6,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def mimic_experiment_breast_primary_hard():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("2.5",)
    MIMIC_FEEDSTOCK.MIN_SEQ_LEN = 5
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    SEQ2SEQ_CONFIG_FEEDSTOCK.L = MIMIC_FEEDSTOCK.MIN_SEQ_LEN
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="kmeans",
        cluster_method_kwargs={"n_clusters": [5]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=6,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG

#   SOFT top diseases
def mimic_experiment_heart_soft_km10():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("7.2",)
    MIMIC_FEEDSTOCK.DIAGNOSIS_SELECTION = "soft"
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    SEQ2SEQ_CONFIG_FEEDSTOCK.TRAIN_STEPS = 100_000
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="kmeans",
        cluster_method_kwargs={"n_clusters": [10]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=0,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def mimic_experiment_urinary_soft_km10():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("10.1",)
    MIMIC_FEEDSTOCK.DIAGNOSIS_SELECTION = "soft"
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    SEQ2SEQ_CONFIG_FEEDSTOCK.TRAIN_STEPS = 100_000
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="kmeans",
        cluster_method_kwargs={"n_clusters": [10]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=0,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def mimic_experiment_gastro_soft_km10():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("9.6",)
    MIMIC_FEEDSTOCK.DIAGNOSIS_SELECTION = "soft"
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    SEQ2SEQ_CONFIG_FEEDSTOCK.TRAIN_STEPS = 100_000
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="kmeans",
        cluster_method_kwargs={"n_clusters": [10]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=0,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def mimic_experiment_kidney_soft_km10():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("10.1.2",)
    MIMIC_FEEDSTOCK.DIAGNOSIS_SELECTION = "soft"
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    SEQ2SEQ_CONFIG_FEEDSTOCK.TRAIN_STEPS = 50_000
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="kmeans",
        cluster_method_kwargs={"n_clusters": [10]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=0,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def mimic_experiment_acs_soft_km10():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("7.2.3", "7.2.4", )
    MIMIC_FEEDSTOCK.DIAGNOSIS_SELECTION = "soft"
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    SEQ2SEQ_CONFIG_FEEDSTOCK.TRAIN_STEPS = 50_000
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="kmeans",
        cluster_method_kwargs={"n_clusters": [10]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=0,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG

def mimic_experiment_heart_hard_km10():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("7.2",)
    MIMIC_FEEDSTOCK.DIAGNOSIS_SELECTION = "hard"
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    SEQ2SEQ_CONFIG_FEEDSTOCK.TRAIN_STEPS = 50_000
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="kmeans",
        cluster_method_kwargs={"n_clusters": [10]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=0,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def mimic_experiment_urinary_hard_km10():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("10.1",)
    MIMIC_FEEDSTOCK.DIAGNOSIS_SELECTION = "hard"
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    SEQ2SEQ_CONFIG_FEEDSTOCK.TRAIN_STEPS = 50_000
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="kmeans",
        cluster_method_kwargs={"n_clusters": [10]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=0,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def mimic_experiment_gastro_hard_km10():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("9.6",)
    MIMIC_FEEDSTOCK.DIAGNOSIS_SELECTION = "hard"
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    SEQ2SEQ_CONFIG_FEEDSTOCK.TRAIN_STEPS = 50_000
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="kmeans",
        cluster_method_kwargs={"n_clusters": [10]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=0,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def mimic_experiment_kidney_hard_km10():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("10.1.2",)
    MIMIC_FEEDSTOCK.DIAGNOSIS_SELECTION = "hard"
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    SEQ2SEQ_CONFIG_FEEDSTOCK.TRAIN_STEPS = 50_000
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="kmeans",
        cluster_method_kwargs={"n_clusters": [10]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=0,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
def mimic_experiment_acs_hard_km10():
    MIMIC_FEEDSTOCK = MIMIC_PROC_ONLY
    MIMIC_FEEDSTOCK.CANCER_CCS_CODE = ("7.2.3", "7.2.4", )
    MIMIC_FEEDSTOCK.DIAGNOSIS_SELECTION = "hard"
    SEQ2SEQ_CONFIG_FEEDSTOCK = CASE_50K_FOR_MIMIC
    SEQ2SEQ_CONFIG_FEEDSTOCK.TRAIN_STEPS = 50_000
    DEFRAG_CONFIG = DefragConfig(
        cluster_method="kmeans",
        cluster_method_kwargs={"n_clusters": [10]},
        cluster_metric="calinski_harabasz",
        exclude=False,
        infer=True,
        nproc=0,
        colour_by="cluster",
    )
    return MIMIC_FEEDSTOCK, SEQ2SEQ_CONFIG_FEEDSTOCK, DEFRAG_CONFIG
