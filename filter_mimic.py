import logging
from pathlib import Path
from pydoc import doc
from typing import Tuple
from matplotlib.scale import scale_factory

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


logging.basicConfig()
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

MIMIC_FEATURES_PATH = Path(__file__).parent.absolute() / "mimic_feature_set.parquet"


def filter_codes_with_tfidf(
    df: pd.DataFrame, 
    vocab_col: str, 
    document_id_col: str, 
    scale_factor: float = 5
) -> pd.DataFrame:
    LOGGER.info(f"Adding tfidf weights for {vocab_col}, split by {document_id_col}.")
    corpus = {}
    doc_ids = df[document_id_col].unique()
    # create pseudo-documents from p/d codes split by document_id_col
    for doc_id in doc_ids:
        _doc_df = df[df[document_id_col] == doc_id]
        # remove null vals
        _doc_df = _doc_df[_doc_df[vocab_col] != ""]
        corpus[doc_id] = {
            "str": " ".join(val for val in _doc_df[vocab_col]),
            "term_series": _doc_df[vocab_col],
            "term_indices": _doc_df[vocab_col].index,
        }
    # Run TfidfVectoriser
    vectorizer = TfidfVectorizer(token_pattern=r'\S+')
    X = vectorizer.fit_transform([corpus[doc_id]["str"] for doc_id in doc_ids])
    tfidf_weights = X.toarray()
    vocab_index_mapping = vectorizer.vocabulary_  # Dict[str, int]
    # append to df a new column with tfidf weights for each token in the corpus
    new_col_name = "tfidf_weights"
    series = []
    indices = []
    for d_idx, doc_id in enumerate(doc_ids):
        weights = tfidf_weights[d_idx, :]
        series += [weights[vocab_index_mapping[term]] for term in corpus[doc_id]["term_series"]]
        indices += corpus[doc_id]["term_indices"].tolist()

    # Add the tf-idf weights to the df
    df.loc[:, new_col_name] = pd.Series(series, index=indices).fillna(value=0)
    # If we want to train using these weights, copy the values to this column name
    # training_col_name = "training_weights"
    # df.loc[:, training_col_name] = df.loc[:, new_col_name]
    # Optional scaling should be applied here
    # df.loc[:, training_col_name].apply(lambda x: 0.00001 if x < 0.35 else x)
    return df


def filter_mimic_features(
    cancer_ccs_code: Tuple,
    min_seq_len: int,
    tfidf_doc_col: str = None,
    tfidf_vocab_col: str = None,
    diagnosis_selection: str = "full", 
    diag_only: bool = False, 
    proc_only: bool = False,
    **kwargs,
) -> pd.DataFrame:
    def annotate_breast_appendix_venn():
        # Add a dummy state_id column. This is a bit of a hack but oh well
        df["state_id"] = "STATE_0"
        # Let's use STATE_ID to colour events by disease type
        for idx, (disease, dids) in enumerate(doc_ids.items()):
            df.loc[df[doc].isin(dids), "state_id"] = f"{disease}_{idx}"
        # And let's append set ops as well
        d1_p_codes = set(df[df.state_id == "2.5_0"].p_icd9_code.unique().tolist())
        d2_p_codes = set(df[df.state_id == "9.6.1_1"].p_icd9_code.unique().tolist())
        I = d1_p_codes & d2_p_codes
        SD = d1_p_codes ^ d2_p_codes
        df["intersection"] = df.p_icd9_code.apply(lambda x: "_I" if x in I else "")
        df["symm_difference"] = df.p_icd9_code.apply(lambda x: "_SD" if x in SD else "")
        df["state_id"] = df.apply(
            lambda x: x["state_id"] + x["intersection"] + x["symm_difference"], 
            axis=1
        )

    def annotate_breast_intra_venn():
        from itertools import chain
        from collections import Counter

        codes = ["15.1", "16.35", "16.37"]
        df.loc[:, "state_id"] = df["p_ccs_lv2"].apply(lambda x: x if x in codes else "Unspecified")
        # And let's append set ops as well
        group_code_sets = dict()
        for code in codes:
            group_code_sets[code] = set(
                df[df.p_ccs_lv2 == code].p_icd9_code.unique().tolist()
            )
        I = set.intersection(*group_code_sets.values())
        SD = set([k for k, v in Counter(chain(*group_code_sets.values())).items() if v==1])  # Chained xor (^)
        df["intersection"] = df.p_icd9_code.apply(lambda x: "_I" if x in I else "")
        df["symm_difference"] = df.p_icd9_code.apply(lambda x: "_SD" if x in SD else "")
        df["state_id"] = df.apply(
            lambda x: x["state_id"] + x["intersection"] + x["symm_difference"], 
            axis=1
        )

    if diag_only and proc_only:
        raise ValueError()
    df = pd.read_parquet(MIMIC_FEATURES_PATH)
    # Rename subject_id to patient_id
    df = df.rename({"subject_id": "patient_id"}, axis=1)

    ccs_keys = {1: "d_ccs_lv1", 2: "d_ccs_lv2", 3: "d_ccs_lv3", 4: "d_ccs_lv4"}

    """
    Note that here we might do some filtering. Even though some of the filtering is based on
    hadm_id, *we always train with the document_id being the subject/patient_id*! The hadm_id is
    only used for filtering out items within the patient's entire sequence. 
    - "full" filtering means that if a patient has a ccs_code of interest *anywhere* in their
      diagnosis list, then that entire patient's seuqence is included.
    - "soft" means the patient's sequence will only include events from hadm_id's where there was
      a ccs_code of interest in the diagnosis list. 
    - "hard" means the patient's sequence will only include events from hadm_id's where there was
      a ccs_code of interest as the first element of the diagnosis list (seq_num == 0). 
    """
    doc = tfidf_doc_col
    if diagnosis_selection in ["soft", "hard"]:
        LOGGER.info(f"Using {diagnosis_selection} filtering.")
        doc = "hadm_id"
    # Find all X cancer patients:
    doc_ids = {}  # A doc id is a patient/subject id or a hadm id
    assert isinstance(cancer_ccs_code, tuple), f"{cancer_ccs_code=} is not a tuple."
    print(f"Before filtering the dataset based on diagnosis codes, there are {len(df)} events.")
    if cancer_ccs_code == ("*",):
        doc_ids[cancer_ccs_code[0]] = df[doc].unique().tolist()
    else:
        for ccs_code in cancer_ccs_code:
            level = ccs_code.count(".") + 1
            # Select only rows which have a diagnosis of interest
            data_filter = (df[ccs_keys[level]] == ccs_code)
            if diagnosis_selection == "hard":
                # Look only at primary diagnosis, not potentially unrelevant diagnoses
                data_filter = data_filter & (df["seq_num"] == 0)
            # If a row meets the requirements of `data_filter`, then 
            # we want to include that entire `doc` in the cohort
            doc_ids[ccs_code] = df[data_filter][doc].to_list()
            print(f"For disease: {ccs_code} there are {len(doc_ids[ccs_code])} patients.")
    # remove patients with more than one of the ccs_keys diseases
    if len(cancer_ccs_code) > 2:
        raise ValueError()
    overlapping_docs = set()
    if len(cancer_ccs_code) > 1:
        overlapping_docs = set.intersection(*list(map(set, doc_ids.values())))
    # filter dataset to include only the patients/admissions of interest
    df = df[df[doc].isin([did for dids in doc_ids.values() for did in dids if did not in overlapping_docs])]
    print(f"After filtering the dataset based on diagnosis codes, there are {len(df)} events.")
    # # Annotate each of the events. This is ad-hoc/bespoke
    # annotate_breast_appendix_venn()
    df["state_id"] = "Unspecified"
    if "2.5" in cancer_ccs_code:
        annotate_breast_intra_venn()

    # Remove all patients who have an unmapped icd code
    patient_ids_not_found = list(
        # Union of these two sets
        set(df[df.d_icd9_code == "NOT FOUND"].patient_id.tolist()) | 
        set(df[df.p_icd9_code == "NOT FOUND"].patient_id.tolist())
    )
    df = df[~df.patient_id.isin(patient_ids_not_found)]
    if diag_only:
        # Keep only the rows with valid diagnosis codes:
        df = df[df.d_icd9_code != ""]
        # Remove the procedure columns
        assert len(list(set(["p_icd9_code", "p_ccs_lv1", "p_ccs_lv2", "p_ccs_lv3"]) & set(kwargs['cols']))) == 0, "Variable mismatch"
        df = df.drop(["p_icd9_code", "p_ccs_lv1", "p_ccs_lv2", "p_ccs_lv3",], axis=1)
    if proc_only:
        # Keep only the rows with valid procedure codes:
        df = df[df.p_icd9_code != ""]
        # Remove the procedure columns
        assert len(list(set(["d_icd9_code", "d_ccs_lv1", "d_ccs_lv2", "d_ccs_lv3"]) & set(kwargs['cols']))) == 0, "Variable mismatch"
        df = df.drop(["d_icd9_code", "d_ccs_lv1", "d_ccs_lv2", "d_ccs_lv3",], axis=1)

    # Remove all patients who recorded fewer than `min_seq_len` events
    vc = df.patient_id.value_counts()
    patient_ids_seq_too_short = vc.index[
        (vc >= min_seq_len) 
        # & (vc < 1024)
    ]
    df = df[df.patient_id.isin(patient_ids_seq_too_short)]

    if tfidf_doc_col is not None and tfidf_vocab_col is not None:
        assert tfidf_doc_col in df.columns, f"{tfidf_doc_col=} not in {df.columns}."
        assert tfidf_vocab_col in df.columns, f"{tfidf_vocab_col=} not in {df.columns}."
        df = filter_codes_with_tfidf(df, vocab_col=tfidf_vocab_col, document_id_col=tfidf_doc_col)

    LOGGER.info(f"Experiment contains {len(df)} events and {len(df.patient_id.unique())} patients.")
    # if len(df) > 60_000:
    #     frac = 60_000 / len(df)
    #     pids = df.patient_id.unique()
    #     pids = pids[np.random.choice(int(frac * len(pids)), len(pids))]
    #     df = df[df.patient_id.isin(pids)]
    LOGGER.info(f"(after filtering) Experiment contains {len(df)} events and {len(df.patient_id.unique())} patients.")
    return df
