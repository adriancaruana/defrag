from pathlib import Path
import random
import dataclasses
import hashlib
import logging
from typing import Dict, List

import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from matplotlib import pyplot as plt


# Determinism
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


NO_CORRELATION_VALUE = -1


class C2VDataGen(torch.utils.data.IterableDataset):
    def __init__(
            self,
            df: pd.DataFrame,
            cols: List,
            n: int = 1
    ):
        self.df = df
        self.cols = cols
        
        self.df['hash'] = self.df.apply(
            lambda x: hashlib.md5(
                '_'.join(map(str, [x[col] for col in self.cols])).encode('utf-8')
            ).hexdigest(), 
            axis=1,
        )
        self._one_hot_encoding_lut = {
            col: {
                i: self.onehot(idx, len(self.df[col].unique()))
                for idx, i in tqdm(enumerate(self.df[col].unique().tolist()))
            }
            for col in self.cols
        }
        self._encoding_lut = self._one_hot_encoding_lut
        # # This is slow for lots of patients
        # self._encoding_lut = {
        #     col: self.get_encoding_lut(self.get_code_counts(self.df, col))
        #     for col in tqdm(self.cols, total=len(self.cols))
        # }
        self.patients = self.df.patient_id.unique().tolist()
        self.patient_dfs = {
            patient: self.unique_adjacent_rows(self.df[self.df.patient_id == patient])
            for patient in self.patients
        }

    def unique_adjacent_rows(self, _df):
        return _df[_df['hash'].ne(_df['hash'].shift())]

    def i_shape(self):
        return {
            col: len(list(self._encoding_lut[col].values())[0])
            for col in self.cols
        }

    @staticmethod
    def get_code_counts(df, var):
        # Get patientwise DFs, and a list of unique codes
        patientwise_dfs = [df[df.patient_id == pid][var] for pid in df.patient_id.unique()]
        codes = df[var].unique().tolist()
        # Get a shifted DF of codes (forward and backward)
        shifted = pd.concat([
            pd.concat((
                patient_df, patient_df.shift(1, fill_value=None).rename("FWD"), 
                patient_df.shift(-1, fill_value=None).rename("BWD")
            ), axis=1)
            for patient_df in patientwise_dfs
        ], axis=0)
        # Count the adjacent codes, for each code
        code_counts = {}
        empty_counts = {code: 0 for code in codes}
        for code in codes:
            value_counts = pd.concat(
                (shifted[shifted[var] == code]["FWD"], shifted[shifted[var] == code]["BWD"]), 
                axis=0
            ).value_counts()
            value_counts = pd.Series(value_counts.values, index=value_counts.index.map(int))
            counts = {**empty_counts, **{_code: _count for _code, _count in zip(value_counts.index, value_counts.values)}}
            code_counts[code] = counts
        return pd.DataFrame(code_counts)

    @staticmethod
    def get_encoding_lut(code_counts_df):
        ordered_codes = code_counts_df.index
        ordered_indices = list(range(len(code_counts_df)))
        numpy_df = np.log(code_counts_df.to_numpy())
        numpy_df[numpy_df == float('-inf')] = NO_CORRELATION_VALUE
        return {
            code: numpy_df[index, :]
            for code, index in zip(ordered_codes, ordered_indices)
        }

    @classmethod
    def onehot(cls, i, l):
        _oh = np.zeros((l))
        _oh[i] = 1.
        return _oh
    
    def random_patient(self):
        return random.choice(self.patients)
    
    def encode(self, col, val):
        return self._encoding_lut[col][val]

    def single_gen(self):
        while True:
            patient_df = self.patient_dfs[self.random_patient()]
            n_events = len(patient_df)
            
            _i0 = random.choice(range(n_events))
            yield {
                col: self._encoding_lut[col][patient_df[col].iloc[_i0]]
                for col in self.cols
            }

    def pair_gen(self):
        while True:
            patient_df = self.patient_dfs[self.random_patient()]
            n_events = len(patient_df)
            
            _i0 = random.choice(range(n_events))
            _i1 = _i0 + 1
            _i1 = 0 if _i1 == n_events else _i1
            
            i0 = {
                col: self._encoding_lut[col][patient_df[col].iloc[_i0]]
                for col in self.cols
            }
            i1 = {
                col: self._encoding_lut[col][patient_df[col].iloc[_i1]]
                for col in self.cols
            }
            
            yield i0, i1

    def __iter__(self):
        # All workers will do the same work, so there is no necessary sharding to do
        if self.n == 1:
            return self.single_gen()
        if self.n == 2:
            return self.pair_gen()
        raise NotImplementedError()



class S2SDataGen(torch.utils.data.IterableDataset):
    """This class produces sequences from the event dataset by:
        1: Selecting a patient at random
        2: Filter the dataset by the selected patient
        3: Take a sequence of L consequtive samples
        4: Encoding each event in the sequence with ENCODER (dim: M)

    The output is an (L, B, M) sequence of embedded events.
    """

    def __init__(
            self,
            df: pd.DataFrame,
            cols: List,
            inherit_encoding_lut: Dict,
            L: int,
            M: int,
            device,
            batch_size: int = 1,
            p_mask_thresh: float = 0.1,
    ):
        super(S2SDataGen).__init__()
        self.df = df
        self.cols = cols
        self.L = L
        self.M = M
        self.batch_size = batch_size
        self.p_mask_thresh = p_mask_thresh
        self.device = device
        
        self._patient_list = self.df.patient_id.unique().tolist()
        self._patient_dfs = {
            patient_id: self.df[self.df.patient_id == patient_id]
            for patient_id in self._patient_list
        }

        # self.remove_short_sequences()
        # # Remove patient sequences that are not long enough to train with. 
        # patient_sequences_to_delete = []
        # for k, v in self._patient_dfs.items():
        #     if self.L is not None and len(v) <= self.L:
        #         patient_sequences_to_delete.append(k)
        # for k in patient_sequences_to_delete:
        #     del self._patient_dfs[k]
        #     del self._patient_list[self._patient_list.index(k)]
        if inherit_encoding_lut is not None:
            print("Inheriting encoding_lut")
            self._encoding_lut = inherit_encoding_lut
        self._one_hot_lut = {
            col: {
                i: self.onehot(idx, len(self.df[col].unique()))
                for idx, i in tqdm(enumerate(self.df[col].unique().tolist()))
            }
            for col in self.cols
        }

        self.code_count_col = {col: {code: (df[col] == code).sum() for code in df[col].unique()} for col in self.cols}
        self.code_count_max_col = {col: max(self.code_count_col[col].values()) for col in self.cols}
        self.code_weight_col = {
            col: {
                code: self.code_count_max_col[col] / self.code_count_col[col][code]  for code in df[col].unique()
            } 
            for col in self.cols
        }
        self.weight_lut = self.get_weight_lut()

        print(f"Embedding patient sequences")
        self._embedded_patient_sequences = {
            patient_id: self._embed_patient_sequence(self._patient_dfs[patient_id])
            for patient_id in tqdm(self._patient_list)
        }
        if "training_weights" in self.df.columns:
            self._weighted_patient_sequences = {
                patient_id: torch.from_numpy(self.df[self.df.patient_id == patient_id].tfidf_weights.to_numpy())
                for patient_id in tqdm(self._patient_list)
            }
        else:
            self._weighted_patient_sequences = {
                patient_id: torch.ones(len(self.df[self.df.patient_id == patient_id]))
                for patient_id in tqdm(self._patient_list)
            }
        self._coded_patient_sequences = {
            patient_id: self._coded_patient_sequence(self._patient_dfs[patient_id])
            for patient_id in tqdm(self._patient_list)
        }

    # def remove_short_sequences(self, num_state_changes: int = 1):
    #     """
    #     Check for and remove any patient sequences which are too short for training
    #     This is a hard limit based on the loss function internals
    #     """
    #     # 1 is for in-between distances in loss fn
    #     # 2 is for the minim number for a valid standard deviation
    #     min_seq_len = num_state_changes + 1 + 2
    #     patients_to_exclude = []
    #     for patient_id, patient_df in self._patient_dfs.items():
    #         if len(patient_df) < min_seq_len:
    #             patients_to_exclude.append(patient_id)
    #     for patient_id in patients_to_exclude:
    #         del self._patient_dfs[patient_id]
    #     self._patient_list = [
    #         patient_id for patient_id in self._patient_list if patient_id not in patients_to_exclude
    #     ]

    def i_shape(self):
        return {
            col: len(list(self._one_hot_lut[col].values())[0])
            for col in self.cols
        }

    @classmethod
    def onehot(cls, i, l):
        _oh = np.zeros((l))
        _oh[i] = 1.
        return _oh

    def get_weight_lut(self):
        weight_lut = {}
        for col in self.cols:
            code_weights = self.code_weight_col[col]
            weight_lut_col = np.zeros((len(code_weights)))
            for code, weight in code_weights.items():
                one_hot = self.encode_col(col, code)
                weight_lut_col[one_hot.astype(bool)] = weight            
            weight_lut[col] = torch.Tensor(weight_lut_col)
        return weight_lut

    def encode_col(self, col, val):
        return self._one_hot_lut[col][val]
    
    def encode_row(self, row):
        return {
            col: self.encode_col(col, row[col])
            for col in self.cols
        }

    def _embed_patient_sequence(self, seq_df: pd.DataFrame):
        # seq_enc_one_hot = [self.encode_row(row) for _, row in seq_df.iterrows()]
        # seq_code_weights = [{col: self.code_weight_col[col][row[col]] for col in self.cols} for _, row in seq_df.iterrows()]
        original_seq_len = len(seq_df)
        padded_df = seq_df.copy().reset_index(drop=True, inplace=False)
        if self.L is not None and original_seq_len < self.L:
            padded_df = padded_df.reindex(range(self.L)).fillna(seq_df.iloc[-1], downcast='infer')
        seq_enc = [
            {col: torch.Tensor(self._encoding_lut[col][row[col]]).to(self.device) for col in self.cols}
            for _, row in padded_df.iterrows()
        ]
        return seq_enc
        # seq_emb = [self.encoder._encode(enc_row) for enc_row in seq_enc]
        # emb_tensors = list(map(lambda x: torch.reshape(x, (1, self.M)), seq_emb))
        # return (
        #     torch.cat(emb_tensors, dim=0).detach().cpu().numpy(),  # These are the input embeddings
        #     # seq_enc_one_hot,  # These are the columnwise onehot labels
        #     # seq_code_weights,  # These are the frequency-weighted codes for each variable (for weighting loss)
        # )

    def _coded_patient_sequence(self, seq_df: pd.DataFrame):
        seq_codes = [
            {col: row[col] for col in self.cols} for _, row in seq_df.iterrows()
        ]
        return seq_codes

    def _exhaustive_iter(self):
        for pid in self._patient_list:
            patient_df_len = len(self._patient_dfs[pid])
            patient_sequence_embedding = self._embedded_patient_sequences[pid]
            weighted_patient_sequence  = self._weighted_patient_sequences[pid]
            assert len(patient_sequence_embedding) == patient_df_len, "Length mismatch"
            padding_mask = None
            # patient_sequence_codes = self._coded_patient_sequences[pid]
            # example_sequence_weights = patient_sequence_weights[start_idx:start_idx + self.L]
            yield (
                patient_sequence_embedding[:patient_df_len],
                weighted_patient_sequence[:patient_df_len],
                padding_mask,
                # patient_sequence_codes
                # Convert labels from List of Dicts to Dict of Lists
                # {
                #     col: torch.Tensor(np.asarray([item[col] for item in patient_sequence_onehot])) 
                #     for col in self.cols
                # },
            )

    def _gen(self):
        # 1) Randomly sample a patient
        random_patient = random.choice(self._patient_list)
        patient_df_len = len(self._patient_dfs[random_patient])
        if self.L is None:
            # Use the whole sequence
            padding_mask = torch.full((patient_df_len,), False)
            start_idx = 0
            length = patient_df_len
        else:
            # Use a random contiguous sequence of self.L events
            start_idx = random.choice(range(max(1, patient_df_len - self.L + 1)))
            length = self.L
            # The sequence is always guaranteed to be at least L long, but trailing events might be
            # padding events. Use the original patient df len to determine this and mask out padding
            # rows if necessary
            padding_mask = torch.full((self.L,), False)
            padding_mask[patient_df_len:] = True
            # if patient_df_len < self.L:
            #     print(self._weighted_patient_sequences[random_patient])
            #     raise ValueError()
        patient_sequence_embedding = self._embedded_patient_sequences[random_patient]
        weighted_patient_sequence  = self._weighted_patient_sequences[random_patient]
        # patient_sequence_codes = self._coded_patient_sequences[random_patient]
        example_sequence_embedding = patient_sequence_embedding[start_idx:start_idx + length]
        example_sequence_weights = weighted_patient_sequence[start_idx:start_idx + length]
        example_sequence_weights = torch.cat([
            example_sequence_weights,
            torch.zeros((len(example_sequence_embedding) - len(example_sequence_weights),))
        ])

        # example_sequence_codes = patient_sequence_codes[start_idx:start_idx + self.L]
        # example_sequence_onehot = patient_sequence_onehot[start_idx:start_idx + self.L]
        # seq_mask = torch.rand(self.L) < self.p_mask_thresh
        # example_sequence_weights = patient_sequence_weights[start_idx:start_idx + self.L]
        return (
            example_sequence_embedding,
            example_sequence_weights,
            padding_mask
            # example_sequence_codes,
            # Convert labels from List of Dicts to Dict of Lists
            # {
            #     col: torch.Tensor(np.asarray([item[col] for item in example_sequence_onehot])) 
            #     for col in self.cols
            # },
            # seq_mask,
            # {
            #     col: torch.Tensor(np.asarray([item[col] for item in example_sequence_weights]))
            #     for col in self.cols
            # }
        )

    def _collate_fn(self, *examples):
        return torch.cat(
            list(map(lambda x: x.reshape(self.L, 1, self.M), examples)),
            dim=1
        )

    def batch_gen(self):
        while True:
            yield self._collate_fn(*[self._gen()  for _ in range(self.batch_size)])

    def __iter__(self):
        # All workers will do the same work, so there is no necessary sharding to do
        for _ in range(int(1e10)):
            yield self._gen()
        # return self.batch_gen()

