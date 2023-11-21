import logging
from pathlib import Path
import shutil
import yaml

from joblib import Parallel, delayed
import numpy as np
import networkx as nx
import papermill as pm
from tqdm import tqdm

from _constants import DEFRAG_RESULTS, DEFRAG_DATA
from defrag import Eval
from notify import notify
from image_processor import plot
from tqdm_joblib import tqdm_joblib


ROOT = Path("paper_experiments/defrag_thesis_experiments")
# ROOT = Path("paper_experiments/demo_experiment/")
# ROOT = Path("experiments/")
gallery_name = "gallery.ipynb"
gallery_template_name = "gallery_template.ipynb"

logging.basicConfig()
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def is_mimic_experiment(exp_path):
    for file_name in exp_path.iterdir():
        if "mimic_config" in file_name.name:
            return True
    return False


def get_experiments_to_score(exp_path):
    if is_mimic_experiment(exp_path):
        # LOGGER.info(f"{exp_path=} is a mimic experiment, no ground truth available to score inferred graph. Skipping.")
        return
    defrag_data_path = exp_path / DEFRAG_DATA
    defrag_results_path = exp_path / DEFRAG_RESULTS
    if not defrag_data_path.exists():
        # LOGGER.info(f"{defrag_data_path=} doesn't exist yet, skipping.")
        return
    if defrag_results_path.exists():
        # LOGGER.info(f"{defrag_results_path=} already exists, skipping.")
        return
    return exp_path


def score_inferred_graph(exp_path):
    defrag_data_path = exp_path / DEFRAG_DATA
    defrag_results_path = exp_path / DEFRAG_RESULTS
    defrag_data = np.load(defrag_data_path)
    ham = defrag_data["ham"]
    syn_am = np.load(exp_path / "syn_G_adjacency_matrix.npz")['am']
    syn_G_nx = nx.from_numpy_matrix(syn_am.T, create_using=nx.DiGraph)
    inf_G_nx = nx.from_numpy_matrix(ham, create_using=nx.DiGraph)
    defrag_results = Eval(syn_G_nx, inf_G_nx).eval()
    with open(defrag_results_path, "w") as f:
        yaml.dump(defrag_results, f)


# def generate_gallery(exp_path):
#         gallery_path = (exp_path / gallery_name)
#         shutil.copy(gallery_template_name, gallery_path)
#         try: 
#             pm.execute_notebook(
#                 gallery_path,
#                 gallery_path,
#                 parameters=dict(
#                     experiment_path=str(exp_path),
#                     raw=False,
#                     change_root=True,
#                     engine='profiling',
#                 )
#             )
#             return None
#         except Exception as e:
#             raise Exception(f"Processing {exp_path} failed. See stacktrace for reason.") from e 

def post_process_experiment(path):
    plot(path)

if __name__ == "__main__":
    # post-process all experiments
    exp_paths = [
        Path("experiments/synthetic_data_thesis_stlo_rpe_only_clo_hdbscan"),
        Path("experiments/synthetic_data_thesis_stlo_rpe_only_sep_hdbscan"),
        Path("experiments/synthetic_data_thesis_stlo_rpe_only_con_hdbscan"),
    ]
    experiment_list = [sub_experiment for experiment_path in ROOT.iterdir() for sub_experiment in experiment_path.glob("experiment_*")]
    # experiment_list = [sub_experiment for experiment_path in exp_paths for sub_experiment in experiment_path.glob("experiment_*")]
    print(f"Found {len(experiment_list)} total experiments.")
    experiments_to_score = [exp_path for exp_path in experiment_list if get_experiments_to_score(exp_path) is not None]
    print(f"Found {len(experiments_to_score)} experiments to score.")
    with tqdm_joblib(tqdm(desc=f"Scoring {len(experiments_to_score)} experiments", total=len(experiments_to_score))):
        Parallel(n_jobs=3)(
            delayed(score_inferred_graph)(sub_experiment) 
            for sub_experiment in experiments_to_score
        )
    # print("Generating gallery for each experiment")
    # # Generate summary notebook for all experiment groups
    # # for experiment_path in ROOT.iterdir():
    # #     generate_gallery(experiment_path)
    # Parallel(n_jobs=-1)(
    #     # delayed(generate_gallery)(experiment_path)
    #     delayed(post_process_experiment)(experiment_path)
    #     for experiment_path in ROOT.iterdir()
    # )
    # Notify me
    notify(f"Finished post-processing experiments.")

