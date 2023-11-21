from pathlib import Path

from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from tqdm import tqdm

from tqdm_joblib import tqdm_joblib


NA_VALS = ["NOT FOUND", " ", "", np.nan]
NA_COLS = ["icd_9_map", "ccs_lv1","ccs_lv2","ccs_lv3"]

ROOT = Path("/data/mimic-iv/files/mimiciv/1.0/hosp/")


def clean_ccs_df(df):
    df = df.rename(dict(zip(df.columns, map(lambda x: eval(x).strip(), df))), axis=1)
    clean_col = lambda df, col: df[col].apply(lambda x: eval(x).strip())
    for col in ["ICD-9-CM CODE", "CCS LVL 1", "CCS LVL 2", "CCS LVL 3",]:
        df[col] = clean_col(df, col)
    return df


def add_icd9_mapping(df, mapping_df):
    mapping = {}
    for _, row in tqdm(mapping_df.iterrows(), total=len(mapping_df)):
        mapping[row.icd10cm] = str(row.icd9cm)

    series = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        if row.icd_version == 9:
            series.append(row.icd_code)
        if row.icd_version == 10:
            series.append(mapping.get(row.icd_code, "NOT FOUND"))
    return df.assign(icd_9_map=pd.Series(series).values)


def add_ccs_mapping(df, mapping_df):
    mapping = {
        str(row["ICD-9-CM CODE"]): (str(row["CCS LVL 1"]), str(row["CCS LVL 2"]), str(row["CCS LVL 3"]), )
        for _, row in tqdm(mapping_df.iterrows())
    }
    ccs_cols = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        ccs_cols.append(dict(zip(["ccs_lv1","ccs_lv2","ccs_lv3",], mapping.get(row["icd_9_map"], ("", "", "", )))))
    return pd.concat([df, pd.DataFrame(ccs_cols)], axis=1)


def remove_na_subjects(df):
    cols = list(set(df.columns) & set(NA_COLS))
    print(cols)


def subject_feature_set(subject_id: int, admi: pd.DataFrame, proc: pd.DataFrame, diag: pd.DataFrame):
    subject_info = {"subject_id": subject_id}
    _admi_df = admi[admi.subject_id == subject_id]  # all admission info for the subject
    _proc_df = proc[proc.subject_id == subject_id]  # all procedure info for the subject
    _diag_df = diag[diag.subject_id == subject_id]  # all diagnosis info for the subject
    hadm_infos = {
        row.hadm_id: {
            "hadm_idx": idx,
            "hadm_id": row.hadm_id,
            "admission_type": row.admission_type,
            "admission_location": row.admission_location,
        }
        for idx, (_, row) in enumerate(_admi_df.sort_values(["admittime"], ascending=True).iterrows())
    }
    diagnosis_infos = {
        hadm_id: {
            row.seq_num: {
                "d_icd9_code": row.icd_9_map,
                "d_ccs_lv1": row.ccs_lv1,
                "d_ccs_lv2": row.ccs_lv2,
                "d_ccs_lv3": row.ccs_lv3,
            }
            for _, row in _diag_df[_diag_df.hadm_id == hadm_id].sort_values(["seq_num"], ascending=True).iterrows()
        }
        for hadm_id in _admi_df.hadm_id.unique()
    }
    procedure_infos = {
        hadm_id: {
            row.seq_num: {
                "p_icd9_code": row.icd_9_map,
                "p_ccs_lv1": row.ccs_lv1,
                "p_ccs_lv2": row.ccs_lv2,
                "p_ccs_lv3": row.ccs_lv3,
            }
            for _, row in _proc_df[_proc_df.hadm_id == hadm_id].sort_values(["seq_num"], ascending=True).iterrows()
        }
        for hadm_id in _admi_df.hadm_id.unique()
    }
    default_diag_info = {
        "d_icd9_code": '',
        "d_ccs_lv1": '',
        "d_ccs_lv2": '',
        "d_ccs_lv3": '',
    }
    default_proc_info = {
        "p_icd9_code": '',
        "p_ccs_lv1": '',
        "p_ccs_lv2": '',
        "p_ccs_lv3": '',
    }
    subject_feature_set = []
    # for hadm_info in hadm_infos:
        # for ((diag_hadm_idx, diag_info), (proc_hadm_idx, proc_info)) in zip(diagnosis_infos[hadm_info["hadm_id"]].items(), procedure_infos[hadm_info["hadm_id"]].items()):
    for hadm_idx, hadm_info in hadm_infos.items():
        diag_info = diagnosis_infos[hadm_idx]
        proc_info = procedure_infos[hadm_idx]
        _max_seq_num = max(map(len, (diag_info, proc_info)))
        for seq_num in range(_max_seq_num):
            subject_feature_set.append(
                {
                    **subject_info, 
                    **hadm_info, 
                    "seq_num": seq_num,
                    **diag_info.get(seq_num + 1, default_diag_info), **proc_info.get(seq_num + 1, default_proc_info)
                }
            )

    return pd.DataFrame(subject_feature_set)

def _build_feature_set(subject_ids, admi: pd.DataFrame, proc: pd.DataFrame, diag: pd.DataFrame,):
    r = pd.concat([
        subject_feature_set(subject_id, admi, proc, diag)
        for subject_id in subject_ids
    ])
    return r


def build_feature_set(admi: pd.DataFrame, proc: pd.DataFrame, diag: pd.DataFrame, n: int = 12):
    # Build feature set using n processes by assigning 1/n subjects to each process, then concat the result
    subjects = admi.subject_id.unique().tolist()
    subject_id_lists = [subjects[i::n] for i in range(n)]
    with tqdm_joblib(tqdm(total=len(subject_id_lists), bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')) as progress_bar:
        subject_feature_sets = Parallel(n_jobs=n)(
            delayed(
                lambda sid_list: _build_feature_set(sid_list, admi, proc, diag)
            )(subject_id_list) 
            for subject_id_list in subject_id_lists
        )
        return pd.concat(subject_feature_sets)


if __name__ == "__main__":
    # load mimic-iv
    print("Loading Mimic")
    proc = pd.read_csv(ROOT / "procedures_icd.csv.gz")
    diag = pd.read_csv(ROOT / "diagnoses_icd.csv.gz")
    admi = pd.read_csv(ROOT / "../core/admissions.csv.gz")

    # Just one patient
    # proc = proc[proc.subject_id == 19729614]
    # diag = diag[diag.subject_id == 19729614]
    # admi = admi[admi.subject_id == 19729614]

    # Subsample
    # sids = admi.subject_id.unique().tolist()[:1000]
    # admi = admi[admi.subject_id.isin(sids)]
    # proc = proc[proc.subject_id.isin(sids)]
    # diag = diag[diag.subject_id.isin(sids)]

    # load icd10 -> icd9 mapping
    print("Loading icd10 -> icd9 mapping")
    proc_icd_mapping_df = pd.read_csv("/data/icd10pcstoicd9gem.csv")
    diag_icd_mapping_df = pd.read_csv("/data/icd10cmtoicd9gem.csv")

    # Load ccs for icd9
    print("Load ccs for icd9")
    proc_ccs_mapping_df = pd.read_csv("/data/CCS/ICD_9/ccs_multi_pr_tool_2015.csv")  # These seem to be all numeric
    diag_ccs_mapping_df = pd.read_csv("/data/CCS/ICD_9/ccs_multi_dx_tool_2015.csv")  # There seems to be alphanumeric values in here

    # Clean the CCS dfs
    print("Clean the CCS dfs")
    proc_ccs_mapping_df = clean_ccs_df(proc_ccs_mapping_df)
    diag_ccs_mapping_df = clean_ccs_df(diag_ccs_mapping_df)

    # Save the ccs files for use elsewhere
    print("Saving the CCS dfs for use elsewhere")
    diag_ccs_mapping_df.to_parquet("diag_ccs_mapping.parquet", index=False)
    proc_ccs_mapping_df.to_parquet("proc_ccs_mapping.parquet", index=False)

    # Map the icd10 codes to icd9 for both procedures and diagnoses
    print("Map the icd10 codes to icd9 for both procedures and diagnoses")
    proc = add_icd9_mapping(proc, proc_icd_mapping_df)
    diag = add_icd9_mapping(diag, diag_icd_mapping_df)

    # Add the CCS hierarchy to the df
    print("Add the CCS hierarchy to the df")
    proc = add_ccs_mapping(proc, proc_ccs_mapping_df)
    diag = add_ccs_mapping(diag, diag_ccs_mapping_df)

    print("Building feature set")
    feature_set = build_feature_set(admi, proc, diag)

    print("Saving feature set")
    feature_set.to_parquet("mimic_feature_set.parquet", index=False)