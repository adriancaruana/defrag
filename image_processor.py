
import dataclasses
from pathlib import Path
import traceback
from typing import Tuple
import sys

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from plot import cmap40
# from case.case import Experiment
from gallery import view_results, Path
from _constants import DEFRAG_DATA, GTG_PLOT_ALPHA, GTG_SOFT_PLOT_ALPHA
from defrag import Defrag
from code_relevancy import (
    RELEVANCY_DICT,
    DIRECT, 
    RELATED, 
    UNRELATED
)

# 12 colours
# CMAP = cmap12
# CMAP *= 20
# 40 colours
CMAP = cmap40
CMAP *= 20

RELEVANT_CODES_ONLY = False

def img_from_path(path: Path):
    return plt.imread(path)

def plot_gtg_on_axis(experiment_path: Path, axis):
    archive = np.load(experiment_path / DEFRAG_DATA)
    ham = archive["ham"]
    Defrag.plot_gtG_from_nxG(
        nxG=Defrag.am_to_nxG(ham),
        bg_colour=None, 
        axis=axis,
        colour_by="cluster",
    )

def _plot_encodings(
        umap_embeddings: np.ndarray,
        dataset_targets: pd.Series,
        ax: plt.Axes,
        dpi: int = 180,
        alpha: float = 0.2,
        s: int = 5,
        focus: bool = True,
    ):
        def sort_fn(class_label: str):
            if isinstance(class_label, int):
                return class_label
            if class_label == "Unspecified":
                return -1
            if class_label.split('_')[-1].isnumeric():
                return int(class_label.split('_')[-1])
            return ord(class_label.split('_')[-1][-1])  # sort by str
        # Prepare
        ys = {
            _state: umap_embeddings[dataset_targets == _state, 0]
            for _state in dataset_targets.unique().tolist()
        }
        xs = {
            _state: umap_embeddings[dataset_targets == _state, 1]
            for _state in dataset_targets.unique().tolist()
        }
        classes = list(ys.keys())
        classes.sort(key=sort_fn)
        color_list = CMAP
        # Umap can sometimes generate off-center plots. Let's centre it. 
        if focus:
            xqmin, xqmax = np.quantile(umap_embeddings[:, 0], 0.05), np.quantile(umap_embeddings[:, 0], 0.95)
            yqmin, yqmax = np.quantile(umap_embeddings[:, 1], 0.05), np.quantile(umap_embeddings[:, 1], 0.95)
            x5pc, y5pc = (xqmax - xqmin) / .9 * 0.2, (yqmax - yqmin) / .9 * 0.2
            xmin, xmax = xqmin - x5pc, xqmax + x5pc
            ymin, ymax = yqmin - y5pc, yqmax + y5pc
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

        for idx, _class in enumerate(classes):
            if _class == "Unspecified" or _class == -1:
                ax.scatter(ys[_class], xs[_class], label=_class, alpha=alpha / 20, s=s, color='black', rasterized=True)
            else:
                ax.scatter(ys[_class], xs[_class], label=_class, alpha=alpha, s=s, color=color_list[idx], rasterized=True)
        ax.set_aspect('equal', 'datalim')

def hide(ax):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # for spine in ax.spines.values():
    #     spine.set_edgecolor('gray')

def show_ax(ax, color):
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for spine in ax.spines.values():
        spine.set_edgecolor(color)

def get_ccs_description_mapping():
    def clean_description_name(name: str) -> str:
        if name.endswith("]"):
            new_name = name[:name.find("[") - 1]
            # print(name, new_name)
            return new_name
        return name
    _df = pd.read_parquet("/workspaces/defrag/proc_ccs_mapping.parquet")
    cols = ["CCS LVL 1", "CCS LVL 2", "CCS LVL 3"]
    dfs = [_df[[x, x + " LABEL"]] for x in cols]
    dfs = list(map(lambda x: x[0].drop_duplicates([x[1]]), zip(dfs, cols)))
    return {
        row[col]: clean_description_name(row[col + " LABEL"])
        for mapping, col in zip(dfs, cols)
        for _, row in mapping.iterrows()
    }


@dataclasses.dataclass
class SyntheticExperimentProcessor:
    experiment_path: Path

    def plot_syn_graphs(self, df, ax, ax_row):
        print("plot_syn_graphs")
        assert ax.shape[1] == len(df)
        for (raw_idx, row) in df.iterrows():
            idx = raw_idx % len(df)
            ax[ax_row, idx].imshow(img_from_path(df.iloc[((idx // 3) * 3) + 2].G[0]))
            hide(ax[ax_row, idx])
            n, a = row["#S"], row["zipf_a"]
            ax[ax_row, idx].title.set_text(f'$|G| = {n}$, $a = {a}$')
            if idx == 0:
                print("plotting y label")
                ax[ax_row, idx].get_yaxis().set_visible(True)
                ax[ax_row, idx].set(ylabel="$G_{\mathrm{syn}}$")

    def plot_inf_graphs(self, df, ax, ax_row):
        print("plot_inf_graphs")
        assert ax.shape[1] == len(df)
        for (raw_idx, row) in df.iterrows():
            idx = raw_idx % len(df)
            ax[ax_row, idx].imshow(img_from_path(row.GTG[0]))
            hide(ax[ax_row, idx])
            ami = f"AMI={row['AMI']:.3f}"
            iso = row.ISO == "🟩"
            iso_text = f"ISO={iso}"
            siso = row["S⊑I"] == "🟩" or row["I⊑S"] == "🟩"
            siso_text = f"SUB-ISO={siso}" if not iso else ""
            ax[ax_row, idx].get_xaxis().set_visible(True)
            ax[ax_row, idx].set(xlabel="\n".join([ami, iso_text, siso_text]))
            if idx == 0:
                print("plotting y label")
                ax[ax_row, idx].get_yaxis().set_visible(True)
                ax[ax_row, idx].set(ylabel="$G^\prime$")
            if not iso or not siso:
                show_ax(ax[ax_row, idx], 'red')
            else:
                show_ax(ax[ax_row, idx], 'green')

    def plot_encodings(self, df, ax, ax_row):
        print("plot_encodings")
        assert ax.shape[1] == len(df)
        for (raw_idx, row) in df.iterrows():
            idx = raw_idx % len(df)
            raw_data_path = self.experiment_path / row["name"] / "final_representations.npz"
            raw_data = np.load(raw_data_path, allow_pickle=True)
            raw_data = {k: v for k, v in raw_data.items()}
            raw_data["pw_seq_embeddings"] = raw_data["pw_seq_embeddings"].item()
            cluster_labels = pd.read_csv(self.experiment_path / row["name"] / "clustered_events.csv")
            _plot_encodings(
                umap_embeddings=raw_data["embeddings"],
                dataset_targets=cluster_labels["y"],
                ax=ax[ax_row, idx],
                alpha=0.01,
                s=1,
            )
            hide(ax[ax_row, idx])
            if idx == 0:
                print("plotting y label")
                ax[ax_row, idx].get_yaxis().set_visible(True)
                ax[ax_row, idx].set(ylabel="$f_{\mathbf{enc}} \\forall X$\ncolour $= v \in G_{\mathrm{syn}}$")

    def plot_clusterings(self, df, ax, ax_row):
        print("plot_clusterings")
        assert ax.shape[1] == len(df)
        for (raw_idx, row) in df.iterrows():
            idx = raw_idx % len(df)
            raw_data_path = self.experiment_path / row["name"] / "final_representations.npz"
            raw_data = np.load(raw_data_path, allow_pickle=True)
            raw_data = {k: v for k, v in raw_data.items()}
            raw_data["pw_seq_embeddings"] = raw_data["pw_seq_embeddings"].item()
            cluster_labels = pd.read_csv(self.experiment_path / row["name"] / "clustered_events.csv")
            _plot_encodings(
                umap_embeddings=raw_data["embeddings"],
                dataset_targets=cluster_labels["y_hat"],
                ax=ax[ax_row, idx],
                alpha=0.01,
                s=1,
            )
            hide(ax[ax_row, idx])
            if idx == 0:
                print("plotting y label")
                ax[ax_row, idx].get_yaxis().set_visible(True)
                ax[ax_row, idx].set(ylabel="$f_{\mathbf{enc}} \\forall X$\ncolour $= c \in G^\prime$")

    def run(self):
        df = view_results(self.experiment_path, raw=True)
        plt.rcParams['text.usetex'] = True

        df = df.reset_index(drop=True)
        fig, ax = plt.subplots(
            9, 6, 
            figsize=(6*1.5, 7*1.5), 
            dpi=480,
            gridspec_kw={'height_ratios':[2,2,2,2,0.75,2,2,2,2]}
        )
        fig.tight_layout(rect=[0, 0.02, 1, 0.98])
        # plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.subplots_adjust(wspace=0.04, hspace=0.4)

        df1 = df[df["#S"] <=6]
        df2 = df[df["#S"] > 6]
        self.plot_syn_graphs(df1, ax, 0)
        self.plot_encodings(df1, ax, 1)
        self.plot_clusterings(df1, ax, 2)
        self.plot_inf_graphs(df1, ax, 3)
        for _ax in ax[4, :]:
            _ax.axis('off')
            # hide(_ax)
            # _ax.imshow([[np.nan]])
        self.plot_syn_graphs(df2, ax, 5)
        self.plot_encodings(df2, ax, 6)
        self.plot_clusterings(df2, ax, 7)
        self.plot_inf_graphs(df2, ax, 8)

        plt.savefig(self.experiment_path / "synthetic_data_experiment.png", dpi=plt.gcf().dpi)
        plt.show()

@dataclasses.dataclass
class MimicExperimentProcessor:
    experiment_path: Path
    relevant_codes_only: bool = False
    save: bool = True
    save_path: str = None
    fig_size_multiplier: float = 1

    def __post_init__(self):
        global RELEVANT_CODES_ONLY
        self.relevant_codes_only = RELEVANT_CODES_ONLY or self.relevant_codes_only

    def plot_clusterings(self, ax, eoi, s=5, alpha=0.2, focus=True):
        raw_data_path = Path(eoi) / "final_representations.npz"
        raw_data = np.load(raw_data_path, allow_pickle=True)
        raw_data = {k: v for k, v in raw_data.items()}
        raw_data["pw_seq_embeddings"] = raw_data["pw_seq_embeddings"].item()
        cluster_labels = pd.read_csv(Path(eoi) / "clustered_events.csv")
        _plot_encodings(
            umap_embeddings=raw_data["embeddings"],
            dataset_targets=cluster_labels["y_hat"],
            ax=ax,
            alpha=alpha,
            s=s,
            focus=focus
        )
        self.hide(ax)

    def hide(self, ax, border=True, yax=True, xax=True):
        ax.get_xaxis().set_visible(not xax)
        ax.get_yaxis().set_visible(not yax)
        # ax.get_xaxis().set_ticks([])
        # ax.get_yaxis().set_ticks([])
        
        if border:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

    def get_code_cluster_stats_using_tfidf(self, mimic_code_series, clustered_events, tfidf_weights):
        mapping = get_ccs_description_mapping()
        mapping[""] = ""
        assert len(mimic_code_series) == len(clustered_events)
        code_cluster_df = pd.DataFrame({
            "code": mimic_code_series.reset_index(drop=True), 
            "cluster": clustered_events.reset_index(drop=True),
            "tfidf_weights": tfidf_weights.reset_index(drop=True),
        })
        code_cluster_df = code_cluster_df[code_cluster_df.code != ""]
        code_cluster_df["code"] = code_cluster_df["code"].apply(lambda x: f"({x}) {mapping[x]}")
        clusters = code_cluster_df.cluster.sort_values().unique()
        cluster_code_scores = {}
        for cluster_name in clusters:
            cluster_label = cluster_name.split("_")[-1]
            df = code_cluster_df[code_cluster_df["cluster"] == cluster_name]
            df = df.groupby(["code"]).agg(
                code_count=('code', lambda x: x.count()),  # Count how often the code appears in the cluster
                tfidf_weight_sum=('tfidf_weights', np.sum),  # Sum the tfidf weights for each code
            )
            score = df.code_count * df.tfidf_weight_sum
            score = score / score.sum()
            score = score.sort_values(ascending=False)
            cluster_code_scores[cluster_label] = score
        return cluster_code_scores

    # def get_code_cluster_stats_using_frequencies(self, mimic_code_series, clustered_events, *args):
    #     mapping = get_ccs_description_mapping()
    #     mapping[""] = ""
    #     assert len(mimic_code_series) == len(clustered_events)
    #     code_cluster_df = pd.DataFrame({"code": mimic_code_series.reset_index(drop=True), "cluster": clustered_events.reset_index(drop=True)})
    #     code_cluster_df["code"] = code_cluster_df["code"].apply(lambda x: f"({x}) {mapping[x]}")
    #     clusters = code_cluster_df.cluster.sort_values().unique()
    #     cluster_distributions = {
    #         cluster_name.split("_")[-1]: code_cluster_df[code_cluster_df["cluster"] == cluster_name]["code"].value_counts()
    #         for cluster_name in clusters
    #     }
    #     return {
    #         k: v / v.sum()
    #         for k, v in cluster_distributions.items()
    #     }

    def filter_cluster_codes(self, cluster_distribution: pd.Series, top_pc: float = 1., min_prop: float = 0.05) -> pd.Series:
        v = cluster_distribution
        include = (
            ((v.cumsum() < top_pc) | (pd.Series([True] + ([False] * (len(v)-1)))))
            & (v > min_prop)
        )
        return cluster_distribution[include]

    def get_alpha(self, code: str) -> float:
        if not self.relevant_codes_only:
            return 1
        experiment_name = str(self.experiment_path)
        relevancy = None
        for k in RELEVANCY_DICT.keys():
            if k in experiment_name:
                relevancy = RELEVANCY_DICT[k]
        if relevancy is None:
            raise ValueError("Couldn't determine the disease of the experiment from the experiment name.")
        if code == "":
            return 0
        code_str = code.split("(")[1].split(")")[0]
        val = relevancy.get(code_str, UNRELATED)
        if val == DIRECT:
            return 1
        elif val == RELATED:
            return 1
        elif val == UNRELATED:
            return 0.2
        raise ValueError()

    def plot_gtg(self, ax):
        gtg_path = self.experiment_path / GTG_PLOT_ALPHA
        if (self.experiment_path / GTG_SOFT_PLOT_ALPHA).exists():
            print("Using soft GTG plot")
            gtg_path = self.experiment_path / GTG_SOFT_PLOT_ALPHA
        ax.imshow(img_from_path(gtg_path))
        hide(ax)

    def run(self):
        self.experiment_path  # This is a sub-experiment dir
        cluster_labels = pd.read_csv(self.experiment_path / "clustered_events.csv")

        mimic_dataset_path = self.experiment_path / "filtered_mimic_features.parquet"
        mimic_dataset = pd.read_parquet(mimic_dataset_path)

        stats = self.get_code_cluster_stats_using_tfidf(
            mimic_dataset["p_ccs_lv2"], cluster_labels["y_hat"], mimic_dataset["tfidf_weights"]
        )
        # stats = self.get_code_cluster_stats_using_frequencies(
        #     mimic_dataset["p_ccs_lv2"], cluster_labels["y_hat"], mimic_dataset["tfidf_weights"]
        # )
        stats = {k: self.filter_cluster_codes(v) for k, v in stats.items()}

        nrows, ncols = 2, 2
        multiplier = len(stats) // 5
        multiplier *= self.fig_size_multiplier
        fig_size = tuple(map(lambda x: multiplier*3*x, (ncols, nrows)))
        fig, ax = plt.subplots(
            ncols=ncols, 
            nrows=nrows, 
            figsize=fig_size,
            dpi=240,
    #        gridspec_kw={'width_ratios':[1] + [2.5 for _ in range(ncols - 1)]}
        )

        gs = ax[0, 0].get_gridspec()
        for _ax in ax[:, 0]:
            _ax.remove()
            # for __ax in _ax:
            #     __ax.remove()
        dist_axis = fig.add_subplot(gs[:, 0])

        colour_list = CMAP

        full_stats = pd.concat(list(map(lambda x: pd.concat((x, pd.Series([np.nan], index=[""]))), stats.values())))[:-1]
        clusters = [cl if idx < len(df) else "" for cl, df in stats.items() for idx in range(len(df) + 1)][:-1]
        full_colours = pd.Series([idx for idx, series in enumerate(stats.values()) for _ in range(len(series) + 1)])[:-1]

        # for idx, code_labels in enumerate(full_stats):
        codes = full_stats.index
        codes = pd.Series(list(f"{code} [{cluster}]" if code != "" else "" for code, cluster in zip(full_stats.index, clusters)))
        codes_ratios = full_stats.values
        y_pos = np.arange(len(codes))
        y_pos_ticks = [idx for idx in np.arange(len(codes)) if pd.notnull(codes_ratios[idx])]
        colours_list = [colour_list[colour_idx] for colour_idx in full_colours]
        alphas_list = [self.get_alpha(code) for code in codes]
        rgba = [(*c, a) for c, a in zip(colours_list, alphas_list)]
        hbars = dist_axis.barh(
            y_pos, 
            codes_ratios, 
            align='center', 
            color=rgba,
            # alpha=,
        )
        dist_axis.set_yticks(y_pos_ticks, labels=codes[codes != ""])
        
        dist_axis.invert_yaxis()  # labels read top-to-bottom
        dist_axis.set_xlabel('Code distribution')
        self.hide(dist_axis, yax=False)

        # Label with specially formatted floats
        dist_axis.bar_label(hbars, fmt='%.2f')
        # dist_axis.set_xlim(right=xlim * 1.5)  # adjust xlim to fit labels

        self.plot_clusterings(ax[0, 1], eoi=self.experiment_path)
        # plot_gtg_on_axis(self.experiment_path, ax[1, 1])GTG_PLOT_ALPHA
        self.plot_gtg(ax[1, 1])
        # gtg_path = self.experiment_path / GTG_PLOT_ALPHA
        # if (self.experiment_path / GTG_SOFT_PLOT_ALPHA).exists():
        #     print("Using soft GTG plot")
        #     gtg_path = self.experiment_path / GTG_SOFT_PLOT_ALPHA
        # ax[1, 1].imshow(img_from_path(gtg_path))
        # hide(ax[1, 1])

        relevant = "_relevant" if self.relevant_codes_only else ""
        if self.save_path is None and self.save:
            plt.savefig(self.experiment_path / f"mimic_intra_diagram{relevant}.png", bbox_inches='tight', dpi=300)
            plt.savefig(self.experiment_path / f"mimic_intra_diagram{relevant}.pdf", bbox_inches='tight', dpi=300)
        elif self.save_path is not None and self.save:
            plt.savefig(self.save_path, dpi=300, bbox_inches='tight')
        plt.show()


def safe_run(cls, path):
    try:
        cls(experiment_path=Path(path)).run()
    except Exception as e:
        print(f"Failed to generate {cls.__name__} Plot.\n{str(e)}")
        print(traceback.format_exc())


def plot(experiment_path):
    if "mimic" in str(experiment_path):
        # Need to plot for each sub-experiment
        for subdir in experiment_path.glob("experiment_*"):
            safe_run(MimicExperimentProcessor, Path(subdir))
    else:
        safe_run(SyntheticExperimentProcessor, Path(experiment_path))


if __name__ == "__main__":
    RELEVANT_CODES_ONLY = True
    # get the experiment name from the first positional argument
    experiment_name = sys.argv[1]
    plot(Path(experiment_name))
