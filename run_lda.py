from pathlib import Path
from itertools import product
import yaml

import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.decomposition import PCA, TruncatedSVD, NMF, LatentDirichletAllocation
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.feature_extraction.text import CountVectorizer
from joblib import Parallel, delayed
from tqdm import tqdm
from gensim.models import Word2Vec


from defrag import Defrag, Eval, _gs_cluster, safe_pairwise_distances
from tqdm_joblib import tqdm_joblib


ROOT = Path(".")

def load(exp_path, experiment_type: str = "mimic"):
    if experiment_type == "mimic":
        data = pd.read_parquet(exp_path.rglob("*.parquet").__next__())
    elif experiment_type == "synthetic":
        data = pd.read_csv(exp_path.rglob("syn_*.csv").__next__())

    return data

def get_defrag_ami(exp_path):
    # Load the clustered stats from exp_path / "clustered_stats.yaml
    with open(exp_path / "clustered_stats.yaml") as f:
        stats = yaml.safe_load(f)
    return stats["ami"]

def hdbscan_cluster(X, labels):
    Xdm = safe_pairwise_distances(X)
    param_dict = {
        "min_cluster_size": list(range(5, 50, 5)) + list(range(50, 800, 50)), 
        "min_samples": [2, 5, 10, 50, 100]
    }
    param_list = [dict(zip(param_dict, v)) for v in product(*param_dict.values())]
    # X, Xdm, targets, cluster_method: str, params, exclude: bool = False, infer: bool = False
    search_results = [
        _gs_cluster(X, Xdm, labels, "hdbscan", params, exclude=False, infer=True)
        for params in tqdm(param_list, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
    ]
    search_results = list(filter(lambda x: x is not None, search_results))
    search_results = pd.DataFrame(search_results).sort_values(["relative_validity"], ascending=False)
    y = _gs_cluster(X, Xdm, labels, "hdbscan", search_results.iloc[0]["params"], exclude=False, infer=True)["infer_clusters"]
    return y


def get_corpus(data, feature_col, label_col: str = None):
    corpus = []
    non_window_corpus = []
    labels = []
    corpus_patient_ids = []
    window_len = 2
    patients = data.patient_id.unique()
    for pid in patients:
        _df = data[data.patient_id == pid].reset_index(drop=True)
        for anchor in range(len(_df)):
            _min = max(0, anchor - window_len)
            _max = min(len(_df), anchor + window_len)
            _df_window = _df.iloc[_min:_max]
            _corpus = " ".join(list(map(str, _df_window[feature_col].map(lambda x: f"A{x}"))))
            if "VAR_1" in _df_window.columns:
                _corpus += " " + " ".join(list(map(str, _df_window["VAR_1"].map(lambda x: f"B{x}"))))
            corpus.append(_corpus)
            if label_col is not None:
                labels.append(_df[label_col].iloc[anchor])
            non_window_corpus.append(f"A{_df[feature_col].iloc[anchor]}")
            corpus_patient_ids.append(pid)

    return corpus, non_window_corpus, labels, corpus_patient_ids

def run_lda(corpus, n_topics, n_iter):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    vocab = vectorizer.get_feature_names_out()
    lda = LatentDirichletAllocation(
        n_components=n_topics, 
        max_iter=n_iter, 
        learning_method="online",
        random_state=42,
        # verbose=1,
        n_jobs=1,
    )
    y = lda.fit_transform(X)
    return y, vocab    

def run_nmf(corpus, n_topics, n_iter):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    vocab = vectorizer.get_feature_names_out()
    nmf = NMF(
        n_components=n_topics, 
        # max_iter=n_iter, 
        random_state=42,
        # verbose=1,
    )
    y = nmf.fit_transform(X)
    return y, vocab    

def run_pca(corpus, labels, n_topics, n_iter):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    vocab = vectorizer.get_feature_names_out()
    pca_features = PCA(
        n_components=10, 
        random_state=42,
    ).fit_transform(np.asarray(X.todense()))
    # y = AgglomerativeClustering(
    #     n_clusters=n_topics,
    # ).fit_predict(pca_features)
    X = np.asarray(pca_features)
    y = hdbscan_cluster(X, labels)

    return y, vocab

def run_hmm_no_window(data, priors='random', seed=42):
    observations = []
    state_labels = []
    lengths = []
    for pid in data.patient_id.unique():
        _df = data[data.patient_id == pid].reset_index(drop=True)
        for _, row in _df.iterrows():
            _observation = [f"A{row.VAR_0}"]
            if "VAR_1" in _df.columns:
                _observation.append(f"B{row.VAR_1}")
            observations.append(_observation)
            state_labels.append(row.state_id)
        lengths.append(len(_df))
    
    states = pd.Series(state_labels).unique().tolist()
    id2state = dict(zip(range(len(states)), states))
    vocabulary = pd.Series([word for sentence in observations for word in sentence]).unique().tolist()
    n_topics = len(states)
    n_trials = len(observations[0])

    # Convert "sentences" to numbers:
    vocab2id = dict(zip(vocabulary, range(len(vocabulary))))
    def sentence2counts(sentence):
        ans = []
        for word, idx in vocab2id.items():
            count = sentence.count(word)
            ans.append(count)
        return ans

    X = []
    for sentence in observations:
        row = sentence2counts(sentence)
        X.append(row)

    data = np.array(X, dtype=int)

    if priors == "random":
        model = hmm.GaussianHMM(
            n_components=n_topics,
            init_params='stmc',
            random_state=seed,
        )
        model.n_features = len(vocabulary)
    elif priors == "uniform":
        model = hmm.GaussianHMM(
            n_components=n_topics,
            init_params='',
            random_state=seed,
        )
        model.n_features = len(vocabulary)
        # Uniform prior on states
        model.startprob_ = np.array([1 / n_topics] * n_topics)
        # Uniform prior on transitions
        model.transmat_ = np.array([[1 / n_topics] * n_topics] * n_topics)
        # Uniform prior on emissions
        model.emissionprob_ = np.array([[1 / model.n_features] * model.n_features] * n_topics)

    model.fit(data, lengths)
    try:
        logprob, y_hat = model.decode(data)
    except:
        # Sometimes the model fails to converge, so we try again with a different seed
        return run_hmm_no_window(data, priors=priors, seed=seed + 1)
    return y_hat, vocabulary

def run_word2vec(corpus, non_window_corpus, labels, n_topics, n_iter):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    vocab = vectorizer.get_feature_names_out()
    corpus = [sentence.split() for sentence in corpus]
    model = Word2Vec(
        sentences=corpus,
        vector_size=10,
        window=5,
        min_count=1,
        workers=4,
        compute_loss=True,
    )
    model.build_vocab(corpus_iterable=corpus)
    model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)

    X = np.array([model.wv[event] for event in non_window_corpus])
    y = hdbscan_cluster(X, labels)

    # y = AgglomerativeClustering(
    #     n_clusters=n_topics,
    # ).fit_predict(np.array([model.wv[event] for event in non_window_corpus]))
    return y, vocab

def run_random(corpus, n_topics):
    return np.random.randint(0, n_topics, len(corpus)), None

def score(y, y_hat, corpus_patient_ids, syn_ham, name: str):
    pw_est_df = pd.DataFrame({"est_targets": y_hat, "patient_id": corpus_patient_ids})
    pw_targets = {pid: pw_est_df[pw_est_df.patient_id == pid]["est_targets"] for pid in pw_est_df.patient_id.unique()}
    ram = Defrag.get_raw_adjacency_matrix(pw_targets, pw_est_df.est_targets.unique().tolist())
    sam = Defrag.get_soft_adjacency_matrix(ram)
    ham = Defrag.get_hard_adjacenct_matrix(sam, threshold=0.2)
    eval_results = Eval(
        syn_G_nx=Defrag.am_to_nxG(syn_ham),
        inf_G_nx=Defrag.am_to_nxG(ham),
    ).eval()
    return {
        f"{name}_" + str(k): v for k, v in 
        {
            "ami": adjusted_mutual_info_score(y, y_hat),
            **eval_results
        }.items()
    }

def run(exp_path, steps: int = 100):
    if "mimic" in str(exp_path):
        experiment_type = "mimic"
        col = "p_icd9_code"
    else:
        experiment_type = "synthetic"
        col = "VAR_0"
        label_col = "state_id"
        defrag_ami = get_defrag_ami(exp_path)
        syn_info = yaml.safe_load(open(exp_path.rglob("catsyn_config*.yml").__next__()))

    data = load(exp_path, experiment_type)

    # n_topics = 5
    # if "mimic" not in str(exp_path):
    n_topics = data.state_id.unique().__len__()

    syn_ham = np.load(exp_path / "syn_G_adjacency_matrix.npz")["am"]

    corpus, non_window_corpus, labels, corpus_patient_ids = get_corpus(data, col, label_col)

    baseline_results = {}
    if RUN_BASELINES:
        y_ghmm_random, vocab = run_hmm_no_window(data, priors="random")
        ghmm_random_results = score(labels, y_ghmm_random, corpus_patient_ids, syn_ham, "GHMM-Random")

        y_ghmm_uniform, vocab = run_hmm_no_window(data, priors="uniform")
        ghmm_uniform_results = score(labels, y_ghmm_uniform, corpus_patient_ids, syn_ham, "GHMM")

        y_lda, vocab = run_lda(corpus, n_topics=n_topics, n_iter=steps)
        lda_results = score(labels, y_lda.argmax(axis=1), corpus_patient_ids, syn_ham, "LatentDirichletAllocation")

        y_nmf, vocab = run_nmf(corpus, n_topics=n_topics, n_iter=steps)
        nmf_results = score(labels, y_nmf.argmax(axis=1), corpus_patient_ids, syn_ham, "Non-NegativeMatrixFactorization")

        y_pca, vocab = run_pca(corpus, labels=labels, n_topics=n_topics, n_iter=steps)
        pca_results = score(labels, y_pca, corpus_patient_ids, syn_ham, "PCA+Cluster")

        y_w2v, _ = run_word2vec(corpus, non_window_corpus, labels=labels, n_topics=n_topics, n_iter=steps)
        w2v_results = score(labels, y_w2v, corpus_patient_ids, syn_ham, "Word2Vec")

        # y_random, vocab = run_random(corpus, n_topics=n_topics)
        # random_results = score(labels, y_random, corpus_patient_ids, syn_ham, "Random")
        baseline_results = {
            **lda_results,
            **nmf_results,
            **pca_results,
            # **random_results,
            # **ghmm_random_results,
            **ghmm_uniform_results,
            **w2v_results,
        }


    results = {
        "experiment_name": str(exp_path.stem),
        "seed": syn_info["seed"],
        "zipf_a": syn_info["variable_generator"]["variable_kwargs"]["distribution_kwargs"]["a"],
        "states": syn_info["states_generator"]["nb_states"],
        "variables": syn_info["variable_generator"]["nb_variables"],
        "bins": syn_info["variable_generator"]["variable_kwargs"]["n_bins"],
        "n_topics": n_topics,
        "steps": steps,
        **{
            "Defrag_" + str(k): v for k, v in 
            {
                "ami": defrag_ami,
                **yaml.safe_load(open(exp_path / "defrag_results.yaml"))
            }.items()
        },
        **baseline_results,
    }

    return results


RUN_BASELINES = False
N_PROC = 1
# exp_path = Path(ROOT / "paper_experiments/synthetic_benchmarks/synthetic_data_experiment_big_1000_bins")
exp_path = Path(ROOT / "paper_experiments/defrag_thesis_experiments/synthetic_data_thesis_stlo_rpe_only_con_hdbscan")
# exp_path = Path(ROOT / "synthetic_data_experiment_big")

# Run the experiment in parallel for each subdirectory in exp_path
subdirs = [sub_exp_path for sub_exp_path in exp_path.iterdir() if sub_exp_path.is_dir()]
# Keep only the subdirs that have a "defrag_results.yaml"
subdirs = [sub_exp_path for sub_exp_path in subdirs if (sub_exp_path / "defrag_results.yaml").exists()]
if N_PROC not in [0, 1]:
    # parallel
    with tqdm_joblib(tqdm(desc="Running baselines for each experiment", total=len(subdirs))):
        results = Parallel(n_jobs=-1, verbose=1)(
            delayed(run)(sub_exp_path) for sub_exp_path in subdirs
        )
else:
    # series
    results = [run(sub_exp_path) for sub_exp_path in tqdm(subdirs)]

pd.DataFrame(results).to_csv(exp_path / "lda_results_wlgk_2.csv", index=False)
# pd.DataFrame(results).to_csv("lda_results_w2v.csv", index=False)

# for i, sub_exp_path in enumerate(exp_path.iterdir()):
#     # If it's a directory, run the experiment
#     if sub_exp_path.is_dir():
#         results = run(sub_exp_path, steps=5)
#         print(results)
#     if i >= 0:
#         break
    
