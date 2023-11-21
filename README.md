# Defrag: Inferring Treatment Pathways from Patient Records

This repository contains the source code for the paper "_Inferring Treatment Pathways from Patient
Records_". 

Abstract:

> Treatment pathways are step-by-step plans outlining the recommended medical care for specific diseases; they are revised when different treatments are found to improve patient outcomes. Examining health records is an important part of this revision process, but inferring patients’ actual treatments from health data is challenging due to the complex event coding schemes and absence of pathway-related annotations. We introduce Defrag, a method for examining health records to infer the real-world treatment steps for a particular patient group. Defrag learns the semantic and temporal meaning of healthcare event sequences, allowing it to reliably infer treatment steps from complex healthcare data. To our knowledge, Defrag is the first pathway-inference method to utilise a neural network (NN), an approach made possible by a novel self-supervised learning objective. We also introduce a testing and validation framework for pathway inference to characterise and evaluate Defrag's pathway inference ability, and to establish benchmarks. We demonstrate Defrag's effectiveness by identifying best-practice pathway fragments for three cancer types in public healthcare records, and by inferring pathways in synthetic experiments, where it significantly outperforms non-NN-based methods. We also provide open-source code for Defrag and the testing framework are provided to encourage further research in this area.

---

## Resources

What is in this repository:
 - The source code for _Defrag_:
    - The Transformer and associated training code are located in `case/`.
    - The pathway inference code is located in `defrag.py`.
 - The source code for the testing and validation framework, which is located in `catsyn/`
 - Programatic experiment configurations, located in `experiment_configs.py`
 - Source code to run experiments, located in `conductor.py`

What is **not** in this repository:
 - The MIMIC-IV data, since the license prohibits redistribution.
 - Most experiment results, including trained models, loss data, and inferred pathways, since they
   can be reproduced (as described below).
 
## Setup

There are two main ways to run the code:

1. By building a Docker image from the provided `Dockerfile`, or
1. By creating a `conda` environment with the provided `env.yml` file.

All package versions should be pinned to the bugfix version, so, provided you run the code via one
of these two steps, you shouldn't have any problems with package versioning. 

That said, the code was only tested on `Ubuntu 22.04.2 LTS`. YMMV.

## Data

All code is provided for running _Defrag_ on synthetic data. However, if you intend to run the code
on MIMIC-IV, you need to:

1. Acquire access to and download the [MIMIC-IV v1.0](https://physionet.org/content/mimiciv/1.0/) dataset.
2. Download `.csv` tables [ICD-10-CM to ICD-9-CM](https://data.nber.org/gem/icd10cmtoicd9gem.csv)
   and [ICD-10-PCS to ICD-9-PCS](https://data.nber.org/gem/icd10pcstoicd9gem.csv). These tables are
   for mapping between ICD 10 and 9 code versions from the [Centers for Medicare & Medicaid
   Services](https://www.nber.org/research/data/icd-9-cm-and-icd-10-cm-and-icd-10-pcs-crosswalk-or-general-equivalence-mappings).
3. Download the [Multi-Level
   CCS](https://hcup-us.ahrq.gov/toolssoftware/ccs/Multi_Level_CCS_2015.zip) zip archive and extract
   `ccs_multi_dx_tool_2015.csv` and `ccs_multi_pr_tool_2015.csv` tables.
4. Run the `build_mimic_feature_set.py` Python script to generate the `mimic_feature_set.parquet`
   file. *Note: You will need to modify the script to specify the appropriate locations of the
   above resources*.

Once you've completed the above steps, you will be able to run experiments on the MIMIC-IV dataset.

## Running Experiments

To run an experiment, its configuration must first be defined in `experiment_configs.py`

Once defined, the experiment, which is referenced via the string used in the function's name, is run
as follows:

```
python3 conductor.py <experiment_name>
```

Conductor orchestrates the running of a single _Defrag_ experiment on a dataset, including filtering
the generating synthetic data or filtering MIMIC, training the Transformer, clustering encodings,
and inferring the pathway. Experiments should be deterministic w.r.t their configuration and, for the most part,
idempotent.

## Reproducing results

The MIMIC-IV results presented in the paper are specified with the following configurations:

```
mimic_experiment_breast_soft_hierarchical_5
mimic_experiment_lung_soft_hierarchical_5
mimic_experiment_melanoma_soft_hierarchical_5
```

For the synthetic data experiments, you need to first conduct the following experiments: 

```
synthetic_data_experiment_big
synthetic_data_experiment_big_1000_bins
```

This runs _Defrag_ on the experiments. After running these experiments, they need to be scored by
running `postprocess_experiments.py`, followed by running the baselines, which can be done with the scripy
`run_lda.py`. 

Finally, the figures from the paper can be reproduced with the code cells in the
`paper_material.ipynb` notebook. 

---

For questions, please raise an issue or contact the authors.

If you use this work, please cite it using the following citation:

_Citation to be added after the manuscript has been published_.
