
from pathlib import Path
import traceback
from typing import Tuple

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from plot import cmap40
# from case.case import Experiment
from gallery import view_results, Path

# 12 colours
# CMAP = cmap12
# CMAP *= 20
# 40 colours
CMAP = cmap40
CMAP *= 20


def _plot_synthetic_experiments(experiment_path):
    def img_from_path(path: Path):
        return plt.imread(path)

    def _plot_encodings(
            umap_embeddings: np.ndarray,
            dataset_targets: pd.Series,
            ax: plt.Axes,
            dpi: int = 180,
            alpha: float = 0.1,
            s: int = 5,
        ):
            def sort_fn(class_label: str):
                if isinstance(class_label, int):
                    return class_label
                if class_label == "Unspecified":
                    return -1
                if class_label.split('_')[-1].isnumeric():
                    return int(class_label.split('_')[-1])
                return class_label.split('_')[-1]  # sort by str
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
            for idx, _class in enumerate(classes):
                if _class == "Unspecified" or _class == -1:
                    ax.scatter(ys[_class], xs[_class], label=_class, alpha=alpha, s=s, color='black')
                else:
                    ax.scatter(ys[_class], xs[_class], label=_class, alpha=alpha, s=s, color=color_list[idx])
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

    def plot_syn_graphs(df, ax, ax_row):
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

    def plot_inf_graphs(df, ax, ax_row):
        def colour(x):
            if isinstance(x, bool):
                return "green" if x else "red"
            return "green" if x > 0.9 else "red"
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

    def plot_encodings(df, ax, ax_row):
        print("plot_encodings")
        assert ax.shape[1] == len(df)
        for (raw_idx, row) in df.iterrows():
            idx = raw_idx % len(df)
            raw_data_path = ROOT / row["name"] / "final_representations.npz"
            raw_data = np.load(raw_data_path, allow_pickle=True)
            raw_data = {k: v for k, v in raw_data.items()}
            raw_data["pw_seq_embeddings"] = raw_data["pw_seq_embeddings"].item()
            cluster_labels = pd.read_csv(ROOT / row["name"] / "clustered_events.csv")
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

    def plot_clusterings(df, ax, ax_row):
        print("plot_clusterings")
        assert ax.shape[1] == len(df)
        for (raw_idx, row) in df.iterrows():
            idx = raw_idx % len(df)
            raw_data_path = ROOT / row["name"] / "final_representations.npz"
            raw_data = np.load(raw_data_path, allow_pickle=True)
            raw_data = {k: v for k, v in raw_data.items()}
            raw_data["pw_seq_embeddings"] = raw_data["pw_seq_embeddings"].item()
            cluster_labels = pd.read_csv(ROOT / row["name"] / "clustered_events.csv")
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

    ROOT = Path(experiment_path)
    df = view_results(ROOT, raw=True)

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
    plot_syn_graphs(df1, ax, 0)
    plot_encodings(df1, ax, 1)
    plot_clusterings(df1, ax, 2)
    plot_inf_graphs(df1, ax, 3)
    for _ax in ax[4, :]:
        _ax.axis('off')
        # hide(_ax)
        # _ax.imshow([[np.nan]])
    plot_syn_graphs(df2, ax, 5)
    plot_encodings(df2, ax, 6)
    plot_clusterings(df2, ax, 7)
    plot_inf_graphs(df2, ax, 8)

    plt.savefig(experiment_path / "synthetic_data_experiment.png", dpi=plt.gcf().dpi)
    plt.show()

def _plot_mimic_experiment(experiment_path):
    def img_from_path(path: Path):
        return plt.imread(path)

    def plot_inf_graphs(df, ax, ax_row):
        def colour(x):
            if isinstance(x, bool):
                return "green" if x else "red"
            return "green" if x > 0.9 else "red"
        assert ax.shape[1] == len(df)
        for (raw_idx, row) in df.iterrows():
            idx = raw_idx % len(df)
            ax[ax_row, idx].imshow(img_from_path(row.GTG[0]))
            hide(ax[ax_row, idx])
            ami = f"AMI={row['AMI']:.3f}"
            iso = row.ISO == "🟩"
            iso_text = f"ISO={iso}"
            siso = row["S-ISO"] == "🟩"
            siso_text = f"SUB-ISO={siso}"
            ax[ax_row, idx].get_xaxis().set_visible(True)
            ax[ax_row, idx].set(xlabel="\n".join([ami, iso_text, siso_text]))

    def _plot_encodings(
            umap_embeddings: np.ndarray,
            dataset_targets: pd.Series,
            ax: plt.Axes,
            dpi: int = 180,
            alpha: float = 0.1,
            s: int = 5,
        ):
            def sort_fn(class_label: str):
                if isinstance(class_label, int):
                    return class_label
                if class_label == "Unspecified":
                    return -1
                if class_label.split('_')[-1].isnumeric():
                    return int(class_label.split('_')[-1])
                return class_label.split('_')[-1]  # sort by str
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
            for idx, _class in enumerate(classes):
                if _class == "Unspecified" or _class == -1:
                    ax.scatter(ys[_class], xs[_class], label=_class, alpha=alpha, s=s, color='black')
                else:
                    ax.scatter(ys[_class], xs[_class], label=_class, alpha=alpha, s=s, color=color_list[idx])
            legend = plt.legend(loc="upper left")
            for lh in legend.legendHandles:
                lh.set_alpha(1)
            ax.set_aspect('equal', 'datalim')


    def plot_clusterings(ax, eoi):
        raw_data_path = Path(eoi) / "final_representations.npz"
        raw_data = np.load(raw_data_path, allow_pickle=True)
        raw_data = {k: v for k, v in raw_data.items()}
        raw_data["pw_seq_embeddings"] = raw_data["pw_seq_embeddings"].item()
        cluster_labels = pd.read_csv(Path(eoi) / "clustered_events.csv")
        _plot_encodings(
            umap_embeddings=raw_data["embeddings"],
            dataset_targets=cluster_labels["y_hat"],
            ax=ax,
            alpha=0.2,
            s=5,
        )
        hide(ax)
        # if idx == 0:
        #     print("plotting y label")
        #     ax[ax_row, idx].get_yaxis().set_visible(True)
        #     ax[ax_row, idx].set(ylabel="$f_{\mathbf{enc}} \\forall X$\ncolour $= c \in G^\prime$")


    def get_ccs_description_mapping():
        _df = pd.read_parquet("/workspaces/defrag/proc_ccs_mapping.parquet")
        cols = ["CCS LVL 1", "CCS LVL 2", "CCS LVL 3"]
        dfs = [_df[[x, x + " LABEL"]] for x in cols]
        dfs = list(map(lambda x: x[0].drop_duplicates([x[1]]), zip(dfs, cols)))
        return {
            row[col]: row[col + " LABEL"]
            for mapping, col in zip(dfs, cols)
            for _, row in mapping.iterrows()
        }

    def get_code_cluster_stats(mimic_code_series, clustered_events):
        mapping = get_ccs_description_mapping()
        mapping[""] = ""
        assert len(mimic_code_series) == len(clustered_events)
        code_cluster_df = pd.DataFrame({"code": mimic_code_series.reset_index(drop=True), "cluster": clustered_events.reset_index(drop=True)})
        code_cluster_df["code"] = code_cluster_df["code"].apply(lambda x: f"({x}) {mapping[x]}")
        clusters = code_cluster_df.cluster.sort_values().unique()
        cluster_distributions = {
            cluster_name: code_cluster_df[code_cluster_df["cluster"] == cluster_name]["code"].value_counts()
            for cluster_name in clusters
        }
        return {
            k: v / v.sum()
            for k, v in cluster_distributions.items()
        }


    def hide(ax, border=True, yax=True, xax=True):
        ax.get_xaxis().set_visible(not xax)
        ax.get_yaxis().set_visible(not yax)
        # ax.get_xaxis().set_ticks([])
        # ax.get_yaxis().set_ticks([])
        
        if border:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)


    def _plot(code_cluster_stats, eoi):
        nc = len(code_cluster_stats)
        subgrid_res = (nc - (nc % 2)) // 2
        ncols = subgrid_res + 1
        nrows = nc

        fig, ax = plt.subplots(
            ncols=ncols, 
            nrows=nrows, 
            figsize=tuple(map(lambda x: 3*x, (ncols, nrows))),
            dpi=240,
            gridspec_kw={'width_ratios':[1] + [2.5 for _ in range(ncols - 1)]}
        )
        # fig.tight_layout(rect=[0, -0.5, 1, 1])
        if nc > 3:
            gs = ax[1, 2].get_gridspec()
            for _ax in ax[:subgrid_res, 1:subgrid_res+1]:
                for __ax in _ax:
                    __ax.remove()
            scatter_ax = fig.add_subplot(gs[:subgrid_res, 1:subgrid_res+1])
            plot_clusterings(scatter_ax, eoi=eoi)
            gs = ax[1, 2].get_gridspec()
            for _ax in ax[subgrid_res:2*subgrid_res, 1:subgrid_res+1]:
                for __ax in _ax:
                    __ax.remove()
            inf_graph_ax = fig.add_subplot(gs[subgrid_res:2*subgrid_res, 1:subgrid_res+1])
            inf_graph_ax.imshow(img_from_path(eoi / "graph_tool_graph_alpha.png"))
            hide(inf_graph_ax)
        else:
            gs = ax[0, 1].get_gridspec()
            ax[0, 1].remove()
            scatter_ax = fig.add_subplot(gs[0, 1])
            plot_clusterings(scatter_ax, eoi=eoi)
            ax[1, 1].imshow(img_from_path(eoi / "graph_tool_graph_alpha.png"))
            hide(ax[1, 1])

        if (nc % 2) == 1:
            for _ax in ax[-1, 1:]:
                hide(_ax)

        color_list = CMAP

        xlim = max(v for values in code_cluster_stats.values() for v in values)
        plt.subplots_adjust(wspace=0, hspace=0)

        for idx, (_ax, (cluster_label, code_labels)) in enumerate(zip(ax[:, 0], code_cluster_stats.items())):
            n = 12
            codes = code_labels[:n].index
            codes_ratios = code_labels[:n].values
            y_pos = np.arange(len(codes))

            hbars = _ax.barh(y_pos, codes_ratios, align='center', color=color_list[idx])
            _ax.set_yticks(y_pos, labels=codes)
            _ax.invert_yaxis()  # labels read top-to-bottom
            _ax.set_xlabel('Code distribution')

            # Label with specially formatted floats
            _ax.bar_label(hbars, fmt='%.2f')
            _ax.set_xlim(right=xlim * 1.5)  # adjust xlim to fit labels

            # rects = ax.patches
            # for rect, label in zip(rects, codes):
            #     width = rect.get_width()
            #     ax.text(
            #         width, rect.get_y() + rect.get_height() / 2, label, ha="left", va="center"
            #     )
            hide(_ax, yax=False)

        plt.savefig(eoi / "mimic_intra_diagram.png", bbox_inches='tight')
        plt.show()


    EOI = Path(experiment_path)  # This is a sub-experiment dir
    raw_data_path = EOI / "final_representations.npz"
    raw_data = np.load(raw_data_path, allow_pickle=True)
    raw_data = {k: v for k, v in raw_data.items()}
    raw_data["pw_seq_embeddings"] = raw_data["pw_seq_embeddings"].item()

    cluster_labels = pd.read_csv(EOI / "clustered_events.csv")

    # A map of subject_ids and their total sequence lengths, ordered.
    sid_len_map = {sid: embeddings["embeddings"].shape[0] for sid, embeddings in raw_data["pw_seq_embeddings"].items()}

    mimic_dataset_path = EOI / "filtered_mimic_features.parquet"
    mimic_dataset = pd.read_parquet(mimic_dataset_path)
    mimic_dataset

    code_cluster_stats = get_code_cluster_stats(mimic_dataset["p_ccs_lv2"], cluster_labels["y_hat"])
    _plot(code_cluster_stats, EOI)

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

def get_code_cluster_stats(mimic_code_series, clustered_events):
    mapping = get_ccs_description_mapping()
    mapping[""] = ""
    assert len(mimic_code_series) == len(clustered_events)
    code_cluster_df = pd.DataFrame({"code": mimic_code_series.reset_index(drop=True), "cluster": clustered_events.reset_index(drop=True)})
    code_cluster_df["code"] = code_cluster_df["code"].apply(lambda x: f"({x}) {mapping[x]}")
    clusters = code_cluster_df.cluster.sort_values().unique()
    cluster_distributions = {
        cluster_name.split("_")[-1]: code_cluster_df[code_cluster_df["cluster"] == cluster_name]["code"].value_counts()
        for cluster_name in clusters
    }
    return {
        k: v / v.sum()
        for k, v in cluster_distributions.items()
    }

def filter_cluster_codes(cluster_distribution: pd.Series, top_pc: float = 0.8, min_prop: float = 0.05) -> pd.Series:
    v = cluster_distribution
    include = (
        ((v.cumsum() < top_pc) | (pd.Series([True] + ([False] * (len(v)-1)))))
        & (v > min_prop)
    )
    return cluster_distribution[include]


def hide(ax, border=True, yax=True, xax=True):
    ax.get_xaxis().set_visible(not xax)
    ax.get_yaxis().set_visible(not yax)
    # ax.get_xaxis().set_ticks([])
    # ax.get_yaxis().set_ticks([])
    
    if border:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

def _plot_encodings(
        umap_embeddings: np.ndarray,
        dataset_targets: pd.Series,
        ax: plt.Axes,
        dpi: int = 180,
        alpha: float = 0.1,
        s: int = 5,
    ):
        def sort_fn(class_label: str):
            if isinstance(class_label, int):
                return class_label
            if class_label == "Unspecified":
                return -1
            if class_label.split('_')[-1].isnumeric():
                return int(class_label.split('_')[-1])
            return class_label.split('_')[-1]  # sort by str
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
        colour_list = CMAP

        for idx, _class in enumerate(classes):
            if _class == "Unspecified" or _class == -1:
                ax.scatter(ys[_class], xs[_class], label=_class, alpha=alpha, s=s, color='black')
            else:
                ax.scatter(ys[_class], xs[_class], label=_class, alpha=alpha, s=s, color=colour_list[idx])
        legend = ax.legend(loc="upper left")
        for lh in legend.legendHandles:
            lh.set_alpha(1)
        ax.set_aspect('equal', 'datalim')


def plot_clusterings(ax, eoi):
    raw_data_path = Path(eoi) / "final_representations.npz"
    raw_data = np.load(raw_data_path, allow_pickle=True)
    raw_data = {k: v for k, v in raw_data.items()}
    raw_data["pw_seq_embeddings"] = raw_data["pw_seq_embeddings"].item()
    cluster_labels = pd.read_csv(Path(eoi) / "clustered_events.csv")
    _plot_encodings(
        umap_embeddings=raw_data["embeddings"],
        dataset_targets=cluster_labels["y_hat"],
        ax=ax,
        alpha=0.2,
        s=5,
    )
    hide(ax)

def img_from_path(path: Path):
    return plt.imread(path)

def _plot_mimic_experiment_2(experiment_path):
    EOI = Path(experiment_path)  # This is a sub-experiment dir
    cluster_labels = pd.read_csv(EOI / "clustered_events.csv")

    mimic_dataset_path = EOI / "filtered_mimic_features.parquet"
    mimic_dataset = pd.read_parquet(mimic_dataset_path)
    mimic_dataset

    stats = get_code_cluster_stats(mimic_dataset["p_ccs_lv3"], cluster_labels["y_hat"])

    stats = {k: filter_cluster_codes(v) for k, v in stats.items()}

    nrows, ncols = 2, 2
    multiplier = len(stats) // 5
    fig, ax = plt.subplots(
        ncols=ncols, 
        nrows=nrows, 
        figsize=tuple(map(lambda x: multiplier*3*x, (ncols, nrows))),
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
    print(len(full_stats.index))
    print(len(clusters))
    codes_ratios = full_stats.values
    y_pos = np.arange(len(codes))
    y_pos_ticks = [idx for idx in np.arange(len(codes)) if pd.notnull(codes_ratios[idx])]

    hbars = dist_axis.barh(
        y_pos, 
        codes_ratios, 
        align='center', 
        color=[colour_list[colour_idx] for colour_idx in full_colours],
    )
    dist_axis.set_yticks(y_pos_ticks, labels=codes[codes != ""])
    
    dist_axis.invert_yaxis()  # labels read top-to-bottom
    dist_axis.set_xlabel('Code distribution')
    hide(dist_axis, yax=False)

    # Label with specially formatted floats
    dist_axis.bar_label(hbars, fmt='%.2f')
    # dist_axis.set_xlim(right=xlim * 1.5)  # adjust xlim to fit labels

    plot_clusterings(ax[0, 1], eoi=EOI)
    ax[1, 1].imshow(img_from_path(EOI / "graph_tool_graph_alpha.png"))
    hide(ax[1, 1])


    plt.savefig(EOI / "mimic_intra_diagram.png", bbox_inches='tight')
    plt.show()


def plot(experiment_path):
    if "mimic" in str(experiment_path):
        # Need to plot for each sub-experiment
        for subdir in experiment_path.glob("experiment_*"):
            try:
                _plot_mimic_experiment_2(subdir)
            except Exception as e:
                print(f"Failed to generate Mimic Plot.\n{str(e)}")
                print(traceback.format_exc())
    else:
        try:
            _plot_synthetic_experiments(experiment_path)
        except Exception as e:
            print(f"Failed to generate Synthetic Data Plot.\n{str(e)}")
            print(traceback.format_exc())