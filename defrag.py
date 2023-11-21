import ast
import dataclasses
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import logging
from pathlib import Path
import string
import shutil
import time
from typing import Dict, List
from unittest import result
from itertools import product
import warnings

import graph_tool as gt
from graph_tool.draw import graph_draw as gt_graph_draw
from graph_tool.draw import sfdp_layout
import hdbscan
from joblib import Parallel, delayed, Memory
import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms.isomorphism import DiGraphMatcher
from matplotlib import pyplot as plt
from matplotlib import cm
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    pairwise_distances,
    pairwise_distances_chunked,
    adjusted_mutual_info_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
    silhouette_samples,
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.signal import convolve
from scipy.stats import mode
from tqdm import tqdm
from umap import UMAP

from tqdm_joblib import tqdm_joblib
from case.case import Experiment
from plot import cmap12, cmap40
from py_wlgk.wlgk import wlgk


logging.basicConfig()
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

TMP_NP_MEMMAP = None
MP_CONTEXT = get_context("spawn")
# config.THREADING_LAYER = 'threadsafe'


cluster_memory = Memory("/tmp/defrag-gs-cluster")


def _safe_convert(x):
    if type(x) in (int, float, dict):
        return x
    return x.item(0)


def get_ccs_description_mapping():
    _df = pd.read_csv("/workspaces/defrag/proc_ccs_mapping.csv")
    cols = ["CCS LVL 1", "CCS LVL 2", "CCS LVL 3"]
    dfs = [_df[[x, x + " LABEL"]] for x in cols]
    dfs = list(map(lambda x: x[0].drop_duplicates([x[1]]), zip(dfs, cols)))
    return {
        row[col]: row[col + " LABEL"]
        for mapping, col in zip(dfs, cols)
        for _, row in mapping.iterrows()
    }


def _clustering_scores(X, Xdm, targets, clusters, suffix):
    calinski_harabasz_ = calinski_harabasz_score(X, clusters)
    davies_bouldin_ = davies_bouldin_score(X, clusters)
    silhouette_ = silhouette_score(Xdm, clusters, metric='precomputed')
    ami_ = adjusted_mutual_info_score(targets, clusters)
    return {
        "calinski_harabasz_" + suffix: float(calinski_harabasz_),
        "davies_bouldin_" + suffix: float(davies_bouldin_),
        "silhouette_score_" + suffix: float(silhouette_),
        "ami_" + suffix: float(ami_),
    }


def classify_noise(X: np.ndarray, est_targets: pd.Series, n_closest=20, progress=False, impl='iter'):
    """Classify the unspecified points from the HDBSCAN clustering"""
    # Vectorised implementation, faster (2x) but uses lots of memory
    if impl == 'vect':
        clusters = est_targets.to_numpy()
        idx_spec = est_targets[est_targets != -1].index.to_numpy()
        idx_unspec = est_targets[est_targets == -1].index.to_numpy()
        X_spec = X[idx_spec]
        X_unspec = X[idx_unspec]
        dist = pairwise_distances(X_unspec, X_spec)
        n_smallest_dist_indices = np.argpartition(dist, n_closest, axis=-1)[:, :n_closest]
        idx_spec_stack = np.repeat(idx_spec, idx_unspec.shape[0]).reshape(-1, idx_unspec.shape[0]).T
        n_smallest_spec_indices = idx_spec_stack[np.arange(idx_unspec.shape[0])[:, None], n_smallest_dist_indices]
        closest_cluster_indices = np.repeat(clusters, idx_unspec.shape[0]).reshape(-1, idx_unspec.shape[0]).T[np.arange(idx_unspec.shape[0])[:, None], n_smallest_spec_indices]
        closest_clusters = mode(closest_cluster_indices, axis=1)[0].flatten()
        est_targets[idx_unspec] = closest_clusters

    # Iterative implementation, slower but uses little memory
    if impl == 'iter':
        unspec_indices = est_targets[est_targets == -1].index.tolist()
        spec_indices = est_targets[est_targets != -1].index.tolist()
        spec_encodings = X[np.asarray(spec_indices), :]
        spec_targets_ri = est_targets[spec_indices].reset_index()
        choices = {}
        it = tqdm(unspec_indices) if progress else unspec_indices
        for unspec_index in it:
            val = X[unspec_index, :]
            mse = ((spec_encodings - val) ** 2).sum(axis=-1) / val.shape[0]
            idxs = np.argsort(mse)[1:1+n_closest]
            candidates = spec_targets_ri[spec_targets_ri.index.isin(idxs)]
            choices[unspec_index] = candidates.est_target.value_counts().index[0]
        for unspec_index, choice in choices.items():
            est_targets.iloc[unspec_index] = choice

    return est_targets

def safe_pairwise_distances(X):
    if 4*X.shape[0]**2 < 30*1024**3:
        LOGGER.info(
            "Computing pairwise distances in-memory."
        )
        return pairwise_distances(X)
    if 4*X.shape[0]**2 > 0.8*shutil.disk_usage("/").free:
        raise RuntimeError(
            f"The distance matrix won't fit in memory or on disk. "
            f"Space required: {(4*X.shape[0]**2 / 1024**3):.3f} GB."
        )
    # Distance matrix will be greater than size of memory.
    # So use np.mmap, and compute it in chunks
    import time
    identifier = str(hash(str(time.time()**2)))[-10:]  # ensure no two files of the same name
    memmap_filename = f"/tmp/np_dm_memmap_{identifier}.dat"
    LOGGER.info(
        f"Using np.memmap to at {memmap_filename} the distance matrix "
        "since it requires more than the available memory."
    )
    dm = np.memmap(
        memmap_filename,
        dtype='float32', 
        mode='w+', 
        shape=(X.shape[0], X.shape[0])
    )
    ## This method is slower, but I cannot understand why.
    # chunk_size = 100
    # for chunk_idx in tqdm(range(0, X.shape[0], chunk_size)):
    #     n_samples = min(chunk_idx + chunk_size, X.shape[0])
    #     dm[chunk_idx:chunk_idx+n_samples, :] = pairwise_distances(X[chunk_idx:chunk_idx+n_samples, :], X)
    chunker = pairwise_distances_chunked(X, working_memory=0, n_jobs=1)
    for row in tqdm(range(X.shape[0])):
        dm[row, :] = next(chunker)
    dm.flush()
    global TMP_NP_MEMMAP
    TMP_NP_MEMMAP = memmap_filename
    return dm


# @cluster_memory.cache
def _gs_cluster(X, Xdm, targets, cluster_method: str, params, exclude: bool = False, infer: bool = False):
    if not isinstance(targets, pd.Series):
        targets = pd.Series(targets)
    if cluster_method == "hdbscan":
        gs_clusterer = hdbscan.HDBSCAN(
            **params,
            metric='euclidean',
            # metric="precomputed",
            # algorithm="generic",
            gen_min_span_tree=True,
            memory="/tmp/hdbscan-min-span-tree-memory",
        )
        gs_clusterer.fit(X)
        relative_validity = float(gs_clusterer.relative_validity_)
    elif cluster_method == "kmeans":
        gs_clusterer = KMeans(
            **params,
            copy_x=True,
            random_state=42,
        ).fit(X)
    elif cluster_method == "hierarchical":
        gs_clusterer = AgglomerativeClustering(
            **params,
            affinity="euclidean",
            linkage="ward",
        ).fit(X)
    else:
        raise ValueError(f"Unknown cluster_method: {cluster_method}")

    clusters = pd.Series(gs_clusterer.labels_, name="est_target")
    indices = np.asarray(list(range(len(clusters))))
    _Xdm = Xdm

    if len(clusters.unique()) == 1 or len(clusters) == 0:
        # Clustering hasn't worked in this case, so ignore this set of params
        return None
    

    # Clustering stats where unknown clusters are *included* in stats
    # LOGGER.info("Getting include stats.")
    include_stats = _clustering_scores(X, Xdm, targets, clusters, suffix="include")
    include_stats["include_clusters"] = clusters

    # Clustering stats where unknown clusters are *excluded* in stats
    # LOGGER.info("Getting exclude stats.")
    if exclude:
        _df = pd.DataFrame({"y": targets, "y_hat": gs_clusterer.labels_})
        indices_exclude = _df["y_hat"] != -1
        indices_to_delete = _df["y_hat"] == -1
        targets_exclude = targets[indices_exclude]
        clusters_exclude = _df[indices_exclude]["y_hat"]
        _Xdm_exclude = _Xdm.copy()  # This uses too much memory
        _Xdm_exclude = np.delete(_Xdm_exclude, indices_to_delete, axis=0)
        _Xdm_exclude = np.delete(_Xdm_exclude, indices_to_delete, axis=1)
        exclude_stats = _clustering_scores(
            X[indices_exclude, :], _Xdm_exclude, targets_exclude, clusters_exclude, suffix="exclude"
        )
        exclude_stats["exclude_clusters"] = clusters_exclude

    # Clustering stats where we classify every point
    # LOGGER.info("Getting infer stats.")
    if infer:
        clusters_infer = classify_noise(X, clusters.copy(), impl='iter')
        infer_stats = _clustering_scores(X, Xdm, targets, clusters_infer, suffix="infer")
        infer_stats["infer_clusters"] = clusters_infer

    clusters = pd.Series(gs_clusterer.labels_, name="est_target")
    return {
        **include_stats,
        **(exclude_stats if exclude else {}),
        **(infer_stats if infer else {}),
        "num_unknown": int((clusters == -1).sum()),
        "num_clusters_ex_unknown": int(len(clusters[clusters != -1].unique())),
        "params": params,
        "relative_validity": relative_validity if cluster_method == "hdbscan" else None,
    }


@dataclasses.dataclass
class DefragConfig:
    cluster_method: str
    cluster_method_kwargs: Dict[str, List]
    # min_cluster_size_range: List = dataclasses.field(default_factory=lambda x: list(range(100, 700, 50)))
    cluster_metric: str = "calinski_harabasz"
    exclude: bool = False
    infer: bool = True
    nproc: int = -1
    colour_by: str = "position"


@dataclasses.dataclass
class Defrag:
    encodings: np.ndarray
    case_experiment: Experiment
    cluster_on: str = "encodings"
    patient_ids: List = None
    cluster_method: str = None
    cluster_method_kwargs: Dict[str, List] = None
    # min_cluster_size_range: List = dataclasses.field(default_factory=lambda x: list(range(100, 700, 50)))
    cluster_metric: str = "calinski_harabasz"
    exclude: bool = False
    infer: bool = True
    subsample_for_clustering: bool = False
    nproc: int = -1
    colour_by: str = "position"

    def __post_init__(self):
        np.random.seed(1)
        np.random.seed(1)

    @property
    def info(self):
        return {
            "cluster_on": self.cluster_on,
            "exclude": self.exclude,
            "infer": self.infer,
        }

    @staticmethod
    def get_umap_nn_graph(encodings) -> np.ndarray:
        umap = UMAP(
            n_neighbors=100,
            min_dist=0.1,
            n_components=2,
            init="random",
            random_state=42,
            verbose=True,
        )
        umap.fit(encodings)
        return umap.graph_

    def cluster(
        self,
        targets: pd.Series = None,
        classify_noise: bool = True,
    ):
        targets = pd.Series(targets, name="target")
        # embeddings_for_clustering = self.embedding_for_clustering(
        #     self.encodings[self.cluster_on]
        # )
        encodings = self.encodings['encodings']
        assert len(targets) == encodings.shape[0], f"{len(targets)=} != {encodings.shape[0]=}"
        if self.patient_ids is not None:
            raise ValueError()
            LOGGER.info(f"Keeping only {len(self.patient_ids)} patients out of {len(self.encodings['pw_seq_embeddings'])}.")
            pw_indices = {}
            count = 0
            # Get indices for each patient id
            for patient_id, di in self.encodings['pw_seq_embeddings'].items():
                pw_indices[patient_id] = list(range(count, count + di["embeddings"].shape[0]))
                count += di["embeddings"].shape[0]
            # Get indices for pids we want to keep
            indices = []
            for pid in self.patient_ids:
                indices += pw_indices[pid]
            # Filter data on indices
            LOGGER.info(f"{len(indices)} events out of {encodings.shape[0]} left after filtering.")
            encodings = encodings[indices, :]
            targets = targets[indices]

        embeddings_for_clustering = self.encodings[self.cluster_on]
        # param_dict = {
        #     # "min_cluster_size": [100 * i for i in range(1, 20, 2)],
        #     # "min_samples": [10 * i for i in range(1, 20, 2)],
        #     "min_cluster_size": self.min_cluster_size_range,
        #     "min_samples": [1],
        #     # "min_cluster_size": [10 * i for i in range(1, 10)],
        #     # "min_samples": [10 * i for i in range(1, 10)],
        #     # "n_clusters": list(range(3, 20)),
        # }
        # Ensure that param_dict is of the correct type:
        param_dict = self.cluster_method_kwargs
        assert isinstance(param_dict, dict), f"{param_dict=} is not of type `dict`."
        for k, v in param_dict.items():
            assert isinstance(k, str), f"Key {k=} is not of type `str`."
            assert isinstance(v, list), f"Value {v=} is not of type `list`."
        param_list = [dict(zip(param_dict, v)) for v in product(*param_dict.values())]

        # Do param search
        search_results = []
        dm_embeddings_for_clustering = safe_pairwise_distances(embeddings_for_clustering)
        targets_for_clustering = targets

        # total_samples = embeddings_for_clustering.shape[0]
        # n_indices = total_samples
        # indices_for_clustering = list(range(total_samples))
        # if self.subsample_for_clustering:
        #     n_indices = 10_000
        #     # subsample for faster clustering
        #     indices_for_clustering = np.random.choice(list(range(total_samples)), n_indices, replace=False)
        # indices_not_for_clustering = np.asarray([x for x in list(range(total_samples)) if x not in indices_for_clustering])
        # embeddings_for_clustering = embeddings_for_clustering[indices_for_clustering, :]
        # if indices_not_for_clustering.shape[0] > 0:
        #     dm_embeddings_for_clustering = np.delete(dm_embeddings_for_clustering, indices_not_for_clustering, axis=0)
        #     dm_embeddings_for_clustering = np.delete(dm_embeddings_for_clustering, indices_not_for_clustering, axis=1)
        # targets_for_clustering = targets[indices_for_clustering].reset_index(drop=True)

        # Run clustering once, just to cache minimum spanning tree to disk
        # self._gs_cluster(
        #     embeddings_for_clustering,
        #     dm_embeddings_for_clustering,
        #     targets_for_clustering,
        #     param_list[0],
        #     unknown_treatment=self.unknown_treatment,
        #     umap_G=umap_G
        # )
        LOGGER.info(
            f"Performing grid search for the best clustering params. Cluster method: {self.cluster_method}, number of iterations={len(param_list)}."
        )
        if self.nproc not in [0, 1]:
            # parallel
            LOGGER.info(f"Grid search with {self.nproc} processes.")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with tqdm_joblib(tqdm(total=len(param_list), bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')) as progress_bar:
                    search_results = Parallel(n_jobs=self.nproc)(#), require='sharedmem')(
                        delayed(lambda params: _gs_cluster(
                            embeddings_for_clustering,
                            dm_embeddings_for_clustering,
                            targets_for_clustering,
                            self.cluster_method,
                            params,
                            # umap_G=umap_G,
                            exclude=self.exclude,
                            infer=self.infer,
                        ))(params) 
                        for params in param_list
                    )
        else:
            # linear
            LOGGER.info(f"Grid search with 1 process.")
            search_results = []
            for params in tqdm(param_list, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
                search_results.append(_gs_cluster(
                    embeddings_for_clustering,
                    dm_embeddings_for_clustering,
                    targets_for_clustering,
                    self.cluster_method,
                    params,
                    # umap_G=umap_G,
                    exclude=self.exclude,
                    infer=self.infer,
                ))

        search_results = list(filter(lambda x: x is not None, search_results))
        key = self.cluster_metric
        if self.cluster_metric != "relative_validity":
            key += "_infer" if self.infer else "_include"
        reverse = any(x in key for x in ["ami", "calinski", "silhouette", "relative_validity"])
        search_results = pd.DataFrame(search_results)
        search_results = search_results.sort_values([key], ascending=not reverse)
        if "Unspecified" not in targets.unique().tolist():# and len(param_list) == 1 and "n_clusters" in param_list[0]:
            # Sort by the results which are closest to the true number of clusters
            # Only do this for synthetic data experiments (when all targets are specified)
            LOGGER.info(f"Primarily sorting by clusters closest to ground truth.")
            search_results["n_cluster_diff"] = (search_results["num_clusters_ex_unknown"] - len(targets.unique())).abs()
            search_results = search_results.sort_values(["n_cluster_diff", key], ascending=(True, not reverse))

        clusterer_results = _gs_cluster(
            embeddings_for_clustering,
            dm_embeddings_for_clustering,
            targets_for_clustering,
            self.cluster_method,
            search_results.iloc[0]["params"],
            # umap_G=umap_G,
            exclude=self.exclude,
            infer=classify_noise,
        )
        clusters = clusterer_results["infer_clusters"] if classify_noise else clusterer_results["include_clusters"]
        est_targets = pd.Series(clusters, name="est_target")

        # est_targets = pd.Series([-1 for _ in range(total_samples)], name="est_target")
        # est_targets[indices_for_clustering] = clusterer.labels_
        # if classify_noise:
        #     est_targets = self.classify_noise(encodings, est_targets.copy(), progress=True)

        # Give cluster classes meaningful names (instead of simple integers)        
        alphabet_mapping = list(Defrag.alphabet_mapping().values())
        est_targets = est_targets.apply(
            lambda x: "Unspecified" if x == -1 else f"cluster_{alphabet_mapping[x]}"
        )

        classifications = pd.DataFrame([targets, est_targets]).transpose()
        # The clusters don't serialise to yaml, so let's just remove them
        search_results = search_results.to_dict(orient="records")
        for r in search_results:
            for k in ["include_clusters", "exclude_clusters", "infer_clusters"]:
                if k in r.keys():
                    del r[k]
        # In case we memmapp'ed a distance matrix to disk, delete it now.
        global TMP_NP_MEMMAP
        if TMP_NP_MEMMAP is not None:
            Path(TMP_NP_MEMMAP).unlink()
        return classifications, {**search_results[0], **self.info, "other_results": search_results, "key": key, "reverse": reverse}

    def cluster_and_score(
        self,
        classify_noise: bool = False,
        plot_save_path: Path = None,
        plot_on: str = "embeddings",
    ) -> Dict:
        classifications, clusterer_info = self.cluster(
            targets=self.encodings["targets"],
            classify_noise=classify_noise,
        )

        cm, y, y_hat, ami, ami_star = self.case_experiment.score(
            classifications=classifications,
            plot_cm=False,
            plot=True if plot_save_path is not None else False,
            umap_embeddings=self.encodings[plot_on],
            plot_save_path=plot_save_path,
        )
        return {
            "y": y,
            "y_hat": y_hat,
            "cm": cm,
            "ami": ami,
            "ami_star": ami_star,
            "clusterer_info": clusterer_info,
        }

    def clusters_to_df(self, clusters: Dict):
        return pd.DataFrame(
            [
                {"y": y, "y_hat": y_hat}
                for y, y_hat in zip(clusters["y"], clusters["y_hat"])
            ]
        )

    def get_clustered_stats(self, clusters_and_scores: Dict):
        y = pd.Series(clusters_and_scores["y"])
        y_hat = pd.Series(clusters_and_scores["y_hat"])
        stats = {
            "ami": clusters_and_scores["ami"],
            "ami_star": clusters_and_scores["ami_star"],
            "clusterer_info": clusters_and_scores["clusterer_info"],
            "nb_states": len(y.unique()),
            "nb_clusters": len(y_hat.unique()),
        }
        nb_y = len(clusters_and_scores["y"])
        pc_y = {
            f"pc_y_{state_name}": (y == state_name).sum() / nb_y
            for state_name in y.unique()
        }
        pc_y_hat = {
            f"pc_y_hat_{cluster_name}": (y_hat == cluster_name).sum() / nb_y
            for cluster_name in y_hat.unique()
        }
        result_dict = {**stats, **pc_y, **pc_y_hat}
        result_dict = {k: _safe_convert(v) for k, v in result_dict.items()}
        return result_dict

    @staticmethod
    def get_raw_adjacency_matrix(
        pw_est_targets: Dict[str, pd.Series], cluster_names: List
    ):
        LOGGER.info("Getting the raw adjacency matrix.")
        raw_adjacency_matrix = np.zeros(
            [len(cluster_names), len(cluster_names)], dtype=int
        )
        cluster_idx_lut = {v: i for i, v in enumerate(sorted(cluster_names))}
        for patient_id, patient_est_targets in tqdm(pw_est_targets.items()):
            patient_est_targets_shift = patient_est_targets.shift(-1)
            for _from, _to in zip(patient_est_targets, patient_est_targets_shift):
                if pd.isnull(_from) or pd.isnull(_to):
                    continue
                x = cluster_idx_lut[_from]
                y = cluster_idx_lut[_to]
                raw_adjacency_matrix[y, x] += 1

        return raw_adjacency_matrix

    @staticmethod
    def get_soft_adjacency_matrix(m: np.ndarray) -> np.ndarray:
        LOGGER.info("Getting the soft adjacency matrix.")
        am = m - m.T
        sam = am / am.max() * -1
        sam[sam < 0] = 0
        return sam

    @staticmethod
    def get_cut_adjacenct_matrix(
        sam: np.ndarray, threshold: float = 0.5
    ) -> np.ndarray:
        LOGGER.info("Getting the cut adjacency matrix.")
        cam = np.where(sam < threshold, 0, sam)
        return cam

    @staticmethod
    def get_hard_adjacenct_matrix(
        sam: np.ndarray, threshold: float = 0.5
    ) -> np.ndarray:
        LOGGER.info("Getting the hard adjacency matrix.")
        ham = np.where(sam < threshold, 0, 1)
        return ham

    @staticmethod
    def alphabet_mapping() -> Dict[int, str]:
        import string
        from itertools import combinations_with_replacement as cwr
        alphabet = string.ascii_lowercase
        num_to_letter = list(alphabet.upper())
        length = 2
        num_to_letter += ["".join(comb).upper() for comb in cwr(alphabet, length)]
        return {i: v for i, v in enumerate(num_to_letter)}

    @staticmethod
    def am_to_nxG(am: np.ndarray) -> nx.DiGraph:
        num_to_letter = Defrag.alphabet_mapping()
        nxG = nx.from_numpy_matrix(np.matrix(am), create_using=nx.DiGraph)
        # assert len(nxG.nodes()) < 27
        # convert numeric node labels to alphabetic node labels
        nxG = nx.relabel_nodes(
            nxG, {nidx: num_to_letter[nidx] for nidx in nxG.nodes()}
        )
        return nxG

    @staticmethod
    def plot_dag_nx(G: nx.Graph, plot: Path = None):
        fig = plt.figure(figsize=(8, 8))
        fig.tight_layout()
        layout = nx.spring_layout(G)
        alphas = np.array(list(e[2]["weight"] for e in G.edges.data()))
        alphas = np.nan_to_num(alphas, nan=0.0, posinf=0.0, neginf=0.0)
        nx.draw_networkx_nodes(G, pos=layout)
        nx.draw_networkx_edges(G, pos=layout, alpha=alphas)
        nx.draw_networkx_labels(G, pos=layout)
        if plot is not None:
            plt.savefig(plot)
        plt.show()

    @staticmethod
    def nx2gt(G: nx.Graph):
        adj_matrix = nx.convert_matrix.to_numpy_matrix(G)
        gtG = gt.Graph(directed=True)
        gtG.add_vertex(adj_matrix.shape[0])
        # Networkx and graph-tool have opposite edge direction convention
        gtG.properties[("e", "weight")] = gtG.new_edge_property("double")
        edge_info = [
            [a, b]
            for a, b in zip(*adj_matrix.nonzero())
        ]
        edge_weights = [
            float(adj_matrix[a, b])
            for a, b in zip(*adj_matrix.nonzero())
        ]
        gtG.add_edge_list(edge_info)
        for e, w in zip(gtG.edges(), edge_weights):
            gtG.ep["weight"][e] = w
        return gtG

    @staticmethod
    def plot_gtG_from_nxG(nxG, plot: Path = None, bg_colour: str = "white", colour_by: str = "position", axis = None, size = None):
        gtG = Defrag.nx2gt(nxG)
        node_idx_list = list(nxG.nodes())
        num_alpha_mapping = {nidx: aidx for nidx, aidx in enumerate(node_idx_list)}
        Defrag.plot_gtG(gtG, num_alpha_mapping, plot=plot, bg_colour=bg_colour, colour_by=colour_by, axis=axis, size=size)

    @staticmethod
    def plot_gtG(
        gtG, num_alpha_mapping: Dict, plot: Path = None, bg_colour: str = "white", colour_by: str = "position", axis = None, size = None
    ):
        # Draw the graph using graph-tool, because it makes prettier graphs.
        # Colour the nodes according to their start/end node status
        cmap = cm.get_cmap("Pastel1")
        cmap = {
            "r": (*cmap.colors[0], 1.),
            "g": (*cmap.colors[2], 1.),
            "b": (*cmap.colors[1], 1.),
            "y": (*cmap.colors[5], 1.),
        }
        # Colour the nodes according to their colour in the embedding plot
        # colour_list = cmap12
        # colour_list *= 5
        colour_list = cmap40
        colour_list *= 20

        # Create vertex properties
        color = gtG.new_vertex_property("vector<double>")  # colour by position
        plot_color = gtG.new_vertex_property("vector<double>")  # colour by cluster
        alphabetic_idx = gtG.new_vertex_property("string")
        edge_color = gtG.new_edge_property("vector<double>")
        # pen_width = gtG.new_vertex_property("float")
        # add the properties to graph
        gtG.vertex_properties["color"] = color
        gtG.vertex_properties["plot_color"] = plot_color
        gtG.vertex_properties["alphabetic_idx"] = alphabetic_idx
        # gtG.vertex_properties["pen_width"] = pen_width
        # assign a value to that property for each node of that graph
        # Sort graph vertices to align colouring with embedding
        for idx, v in enumerate(list(gtG.vertices())):
        # for v in gtG.vertices():
            # print(v, num_alpha_mapping[v], type(v))#, v.vertex_index, type(v.vertex_index))
            alphabetic_idx[v] = num_alpha_mapping[v]
            plot_color[v] = (*colour_list[idx], 1)  # cluster
            in_e, out_e = list(v.in_edges()), list(v.out_edges())
            if len(in_e) == 0 and len(out_e) != 0:
                # This is a start-node
                color[v] = cmap["g"]
            if len(in_e) != 0 and len(out_e) == 0:
                # This is an end-node
                color[v] = cmap["r"]
            if len(in_e) != 0 and len(out_e) != 0:
                # These are intermediate nodes
                color[v] = cmap["b"]
            if len(in_e) == 0 and len(out_e) == 0:
                # These are isolated nodes
                color[v] = cmap["y"]

        for idx, (e) in enumerate(list(gtG.edges())):
            edge_color[e] = (0, 0, 0, gtG.ep["weight"][e])

        if colour_by == "cluster":
            vertex_fill_color = gtG.vertex_properties["plot_color"]
        elif colour_by == "position":
            vertex_fill_color = gtG.vertex_properties["color"] 
        else:
            raise ValueError()

        gt_graph_draw(
            gtG,
            pos=sfdp_layout(gtG, eweight=gtG.ep["weight"]),
            vertex_text=gtG.vertex_properties["alphabetic_idx"],
            output=str(plot) if plot is not None else None,
            # bg_color=None,
            bg_color=bg_colour,
            # vertex_color=gtG.vertex_properties["plot_color"],
            vertex_fill_color=vertex_fill_color,
            edge_color=edge_color,
            # vertex_pen_width=gtG.vertex_properties["pen_width"],
            fit_view=True,
            output_size=size if size is not None else (1800, 1800),
            adjust_aspect=False,
            mplfig=axis
        )

    @staticmethod
    def plot_adjacency_matrix(m: np.ndarray, tick_labels: List, plot: Path = None):
        assert m.shape[0] == m.shape[1]
        fig = plt.figure(figsize=(3, 3), dpi=240)
        ax = sns.heatmap(m, square=True, cmap="cividis", linewidths=0.1)
        plt.title("Soft adjacency matrix", y=1.08)
        plt.xticks(ticks=np.array(range(len(tick_labels))) + 0.5, labels=tick_labels)
        plt.yticks(ticks=np.array(range(len(tick_labels))) + 0.5, labels=tick_labels)
        plt.xlabel("Node TO")
        plt.ylabel("Node FROM")
        fig.tight_layout()
        if plot is not None:
            plt.savefig(plot)
        plt.show()


@dataclasses.dataclass
class Eval:
    syn_G_nx: nx.DiGraph
    inf_G_nx: nx.DiGraph

    @staticmethod
    def consistent_labelling(G1, G2):
        # Use DiGraphMatcher for directed graphs
        matcher = DiGraphMatcher(G1, G2)
        is_isomorphic = matcher.is_isomorphic()

        if is_isomorphic:
            mapping = matcher.mapping
            inverse_mapping = {v: k for k, v in mapping.items()}
            G2_relabeled = nx.relabel_nodes(G2, inverse_mapping)
            # Reconstruct G2 with nodes in the same order as G1
            # Start by just creating an empty digraph with the same number of nodes as G1
            G2_reconstructed = nx.DiGraph()
            G2_reconstructed.add_nodes_from(G1.nodes())
            # Then add edges from G2_relabeled to G2_recunstructed
            for v in G2.nodes():
                for u in G2.successors(v):
                    G2_reconstructed.add_edge(inverse_mapping[v], inverse_mapping[u])
            return G1, G2_reconstructed
        else:
            raise ValueError("Graphs are not isomorphic")
    
    @staticmethod
    def check_consistent_labelling(G1, G2):
        assert nx.is_isomorphic(G1, G2), "Not Isomorphic"
        for n1, n2 in zip(G1.nodes, G2.nodes):
            assert n1 == n2, f"Node values aren't the same: {n1=}, {n2=}"
            assert set(G1.successors(n1)) == set(G2.successors(n2)), f"Successors aren't the same: {set(G1.successors(n1))=}, {set(G1.successors(n2))=}"
            assert set(G1.predecessors(n1)) == set(G2.predecessors(n2)), f"Predecessors aren't the same:: {set(G1.predecessors(n1))=}, {set(G2.predecessors(n2))=}"

    def eval(self):
        # (1) Are they isomorphic?
        is_iso = DiGraphMatcher(self.syn_G_nx, self.inf_G_nx).is_isomorphic()
        # (2) Is there some form of sub-graph isomorphism?
        syn_contains_inf = None
        inf_contains_syn = None
        if len(self.syn_G_nx.nodes()) >= len(self.inf_G_nx.nodes()):
            syn_contains_inf = DiGraphMatcher(
                self.syn_G_nx, self.inf_G_nx
            ).subgraph_is_isomorphic()
        if len(self.inf_G_nx.nodes()) >= len(self.syn_G_nx.nodes()):
            inf_contains_syn = DiGraphMatcher(
                self.inf_G_nx, self.syn_G_nx
            ).subgraph_is_isomorphic()
        # (3) What is the graph edit distance
        edit_distance = None
        approx_ged = False
        if len(self.syn_G_nx) < 15 and len(self.inf_G_nx) < 15:
            if not is_iso:
                best_cost = None
                upper_bound = len(self.syn_G_nx)
                LOGGER.info(f"Computing graph edit distance. This might take a while.")
                TIMEOUT = 2*60*60  # Two hours
                t = time.time()
                for _, _, cost in nx.optimize_edit_paths(
                    self.syn_G_nx, 
                    self.inf_G_nx, 
                    strictly_decreasing=True, 
                    # upper_bound=upper_bound,
                    timeout=TIMEOUT,
                ):
                    LOGGER.info(f"New lowest cost found: {int(cost)}, (max={upper_bound}).")
                    best_cost = int(cost)
                if time.time() - t > TIMEOUT:
                    LOGGER.info(f"Timeout reached; graph edit distance is approximate.")
                    approx_ged = True
                edit_distance = best_cost
        # (4) Calculate a sorted degree sequence similarity
        deg_dist = None
        if len(self.syn_G_nx) == len(self.inf_G_nx):
            # Get the in and out degree sequence of the directed synthetic graph
            syn_in_deg_seq = sorted([d for n, d in self.syn_G_nx.in_degree()])
            syn_out_deg_seq = sorted([d for n, d in self.syn_G_nx.out_degree()])
            # Get the in and out degree sequence of the directed inferred graph
            inf_in_deg_seq = sorted([d for n, d in self.inf_G_nx.in_degree()])
            inf_out_deg_seq = sorted([d for n, d in self.inf_G_nx.out_degree()])
            # Calculate the euclidean distance between the degree sequences
            in_deg_dist = np.linalg.norm(np.array(syn_in_deg_seq) - np.array(inf_in_deg_seq))
            out_deg_dist = np.linalg.norm(np.array(syn_out_deg_seq) - np.array(inf_out_deg_seq))
            # Calculate the average degree sequence distance
            deg_dist = float((in_deg_dist + out_deg_dist) / 2)

        # (5) Calculate the Weisfeiler-Lehman kernel similarity
        wlgk_val = None
        if len(self.syn_G_nx) < 50 and len(self.inf_G_nx) < 50:
            syn_G = self.syn_G_nx.copy()
            inf_G = self.inf_G_nx.copy()
            if is_iso:
                # Try and label the graphs consistently if they're isomorphic
                syn_G, inf_G = Eval.consistent_labelling(syn_G, inf_G)
                Eval.check_consistent_labelling(syn_G, inf_G)
            # Add node labels
            nx.set_node_attributes(syn_G, "", "node_label")
            nx.set_node_attributes(inf_G, "", "node_label")
            # nodes are alphabetically labelled. Remap them to numerical
            num_to_letter = Defrag.alphabet_mapping()
            letter_to_num = {v: k for k, v in num_to_letter.items()}
            if list(syn_G.nodes())[0] in letter_to_num.keys():
                syn_G = nx.relabel_nodes(syn_G, {k: letter_to_num[k] for k in syn_G.nodes()})
                inf_G = nx.relabel_nodes(inf_G, {k: letter_to_num[k] for k in inf_G.nodes()})
            # Compute the Weisfeiler-Lehman kernel
            wlgk_val = float(wlgk(syn_G, inf_G, num_iterations=1))

        return {
            "is_isomorphic": is_iso,
            "syn_contains_inf": syn_contains_inf,
            "inf_contains_syn": inf_contains_syn,
            "edit_distance": edit_distance,
            "approx_ged": approx_ged,
            "degree_distance": deg_dist,
            "wlgk": wlgk_val,
        }
