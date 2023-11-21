import dataclasses
import logging
from pathlib import Path
from socket import MSG_OOB
from typing import Iterable, List, Dict, Tuple
import random
import itertools
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
# import kmapper as km
import torch
from torch.autograd import Variable
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics.cluster import adjusted_mutual_info_score
from hdbscan import HDBSCAN
import umap

from .data import C2VDataGen, S2SDataGen
from .models import SiameseModel, AEModel, Seq2Seq


# Determinism
torch.manual_seed(1)
np.random.seed(1)
random.seed(1)


STATUS_HANDLER = None
CPU, GPU = torch.device("cpu"), torch.device("cuda:0")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PRELOAD_DATA_ON_GPU = False


def update_status(msg: str, level_attr: str = "info"):
    global STATUS_HANDLER
    if not isinstance(STATUS_HANDLER, tqdm):
        # LOGGER.__getattr__(level_attr)
        LOGGER.info(msg)
        return
    STATUS_HANDLER.set_postfix_str(msg)


def progress_with_status(it: Iterable, total=None):
    global STATUS_HANDLER
    if total:
        ncols = str(min(total, 20))
    else:
        ncols = str(min(len(it), 20))
    barfmt = "{l_bar}{bar:" + ncols + "}{r_bar}{bar:-" + ncols + "b}"
    return (STATUS_HANDLER := tqdm(it, bar_format=barfmt, total=total))


def save_model(model: torch.nn.Module, path: Path):
    torch.save(model.state_dict(), path)


def load_model(model: torch.nn.Module, path: Path):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


@dataclasses.dataclass
class Config:
    model_type: str
    cols: List
    train_steps: int


@dataclasses.dataclass
class Cat2VecConfig(Config):
    n_col_features: int
    n_state_features: int


@dataclasses.dataclass
class Seq2SeqConfig(Config):
    l: int  # Length of events in the sequence
    m: int  # Number of features for each event in the sequence
    n: int  # Number of heads for each Multi-Headded Attention Block
    num_encoder_layers: int
    num_decoder_layers: int
    dim_feedforward: int
    dropout: int
    batch_size: int
    p_mask_thresh: float  # Probability of masking input tokens
    validation_steps: int
    encoder_window_size: int


@dataclasses.dataclass
class Experiment:
    data_path: Path
    train_pc: float = 0.8
    verbose: bool = False

    def __post_init__(self):
        torch.manual_seed(1)
        np.random.seed(1)
        random.seed(1)

        if self.verbose:
            logging.basicConfig()
            logging.getLogger().setLevel(logging.INFO)            
        self.data = self.load_data()

    def load_data(self):
        """Load the data, and do a train/test split based on patient_id's"""
        if "parquet" in str(self.data_path):
            data = pd.read_parquet(self.data_path)
        else:
            data = pd.read_csv(self.data_path)
        patients = data.patient_id.unique().tolist()
        full_data = data[list(filter(lambda x: x != 'state_id', data.columns))]
        full_target = data['state_id']
        full = {'data': full_data, 'target': full_target}
        # Split patient_id's
        train_patients = np.random.default_rng(seed=0).choice(patients, int(np.round(len(patients) * self.train_pc)))
        test_patients = [p for p in patients if p not in train_patients]
        # Train
        train_rows = data[data.patient_id.isin(train_patients)]
        train_data = train_rows[list(filter(lambda x: x != 'state_id', train_rows.columns))]
        train_target = train_rows['state_id']
        train = {'data': train_data, 'target': train_target,}
        # Test
        test_rows = data[data.patient_id.isin(test_patients)]
        test_data = test_rows[list(filter(lambda x: x != 'state_id', test_rows.columns))]
        test_target = test_rows['state_id']
        test = {'data': test_data, 'target': test_target,}
        return {'full': full, 'train': train, 'test': test}

    def _build_cat2vec_model(self, cat2vec_config: Cat2VecConfig, ishape: Dict):
        if cat2vec_config.model_type == "SiameseModel":
            return SiameseModel(
                i_shape_dict=ishape,
                n_col_features=cat2vec_config.n_col_features,
                n_state_features=cat2vec_config.n_state_features,
            )
        if cat2vec_config.model_type == "AEModel":
            return AEModel(
                i_shape_dict=ishape,
                n_col_features=cat2vec_config.n_col_features,
                n_state_features=cat2vec_config.n_state_features,
            )
        raise NotImplementedError()

    def _build_seq2seq_model(self, seq2seq_config: Seq2SeqConfig, ishape_dict):
        if seq2seq_config.model_type == "Seq2SeqTR":
            return torch.nn.Transformer(
                d_model=seq2seq_config.m,
                nhead=seq2seq_config.n,
                num_encoder_layers=seq2seq_config.num_encoder_layers,
                num_decoder_layers=seq2seq_config.num_decoder_layers,
                dim_feedforward=seq2seq_config.dim_feedforward,
                dropout=seq2seq_config.dropout,
                # batch_first=True,
            )
        if seq2seq_config.model_type == "Seq2Seq":
            return Seq2Seq(
                column_cardinality=ishape_dict,
                n_state_features=seq2seq_config.m,
                num_heads=seq2seq_config.n,
                num_encoder_layers=seq2seq_config.num_encoder_layers,
                num_decoder_layers=seq2seq_config.num_decoder_layers,
                feedforward_dim=seq2seq_config.dim_feedforward,
                dropout=seq2seq_config.dropout,
                encoder_window_size=seq2seq_config.encoder_window_size,
            )
        raise NotImplementedError()


    # def _barlow_twins_tuning(
    #     self,
    #     seq2seq_config: Seq2SeqConfig,
    #     seq2seq_model: torch.nn.Module,
    #     dg_train: S2SDataGen,
    #     dg_validation: S2SDataGen,
    #     _lambda: float = 5e-3,
    #     plot_barlow_loss: Path = None,
    # ):
    #     def _barlow_step(X, p_mask_thresh):
    #         L, B, D = X.shape
    #         N = B * L
    #         # two randomly token masks
    #         m_a = torch.stack([torch.rand(L) < p_mask_thresh for _ in range(B)]).to(DEVICE)
    #         m_b = torch.stack([torch.rand(L) < p_mask_thresh for _ in range(B)]).to(DEVICE)
    #         X = X.to(DEVICE)
    #         mask = seq2seq_model.get_mask(L).to(DEVICE)
    #         # compute embeddings
    #         z_a = seq2seq_model(
    #             X, columnwise_onehot_events=None, sequence_mask=m_a.transpose(1, 0)
    #         )['embedded_sequence']
    #         z_b = seq2seq_model(
    #             X, columnwise_onehot_events=None, sequence_mask=m_b.transpose(1, 0)
    #         )['embedded_sequence']

    #         # z_a = seq2seq_model.tr.encoder(X, mask=mask, src_key_padding_mask=m_a.transpose(1, 0))
    #         # z_b = seq2seq_model.tr.encoder(X, mask=mask, src_key_padding_mask=m_b.transpose(1, 0))
    #         z_a = z_a.reshape(N, -1)
    #         z_b = z_b.reshape(N, -1)
    #         # normalize repr. along the batch dimension
    #         z_a_norm = (z_a - z_a.mean(dim=0)) / z_a.std(dim=0)  # NxD
    #         z_b_norm = (z_b - z_b.mean(dim=0)) / z_b.std(dim=0)  # NxD
    #         # cross-correlation matrix
    #         c = z_a_norm.T @ z_b_norm / N # DxD
    #         # loss
    #         eye = torch.eye(D).to(DEVICE)
    #         c_diff = (c - eye).pow(2) # DxD
    #         # multiply off-diagonal elems of c_diff by lambda
    #         c_diff = torch.where(eye.type(torch.bool), c_diff, c_diff * _lambda)
    #         loss = c_diff.sum()
    #         return loss

    #     def get_validation_loss(dg_validation):
    #         with torch.no_grad():
    #             example_loss_li = []
    #             for X, _ in dg_validation._exhaustive_iter():
    #                 X = X.unsqueeze(0).transpose(0, 1)
    #                 loss = _barlow_step(X, p_mask_thresh=dg_validation.p_mask_thresh)
    #                 loss = loss.to(CPU).detach().numpy()
    #                 example_loss_li.append(loss)
    #         return np.mean(example_loss_li)

    #     print("Running Barlow Twins Tuning...")
    #     dl_train = torch.utils.data.DataLoader(
    #         dg_train,
    #         batch_size=seq2seq_config.batch_size,
    #         num_workers=3,
    #     )
    #     params = seq2seq_model.tr.encoder.parameters()
    #     optim = torch.optim.AdamW(
    #         params, lr=0.0001, betas=(0.9, 0.98), eps=1e-9
    #     )
    #     # lambda: weight on the off-diagonal terms
    #     # D: dimensionality of the embeddings
    #     D = seq2seq_config.m
    #     loss_li, running_mean_loss_li, validation_loss_li = [], [], []
    #     for t, (X, _, _) in progress_with_status(
    #         zip(range(int(seq2seq_config.train_steps)), dl_train), 
    #         total=seq2seq_config.train_steps
    #     ):
    #         X = dg_train._collate_fn(*X)
    #         loss = _barlow_step(X, p_mask_thresh=dg_train.p_mask_thresh)
    #         # optimization step
    #         optim.zero_grad()
    #         loss.backward()
    #         optim.step()
    #         # Report on the loss
    #         _loss = loss.to(CPU)
    #         _loss = _loss.detach().numpy()
    #         loss_li.append(_loss)
    #         if len(running_mean_loss_li) == seq2seq_config.train_steps // 10:
    #             running_mean_loss_li = running_mean_loss_li[1:]
    #         running_mean_loss_li.append(_loss)
    #         if t % (seq2seq_config.train_steps // seq2seq_config.validation_steps) == 0:
    #             validation_loss_li.append(get_validation_loss(dg_validation))
    #             update_status(
    #                 f"Mean loss: {np.mean(running_mean_loss_li):.4E}" + 
    #                 ('' if len(validation_loss_li) == 0 else f", Validation Loss: {validation_loss_li[-1]:.4E}") + 
    #                 "."
    #             )
    #     STATUS_HANDLER = None
    #     if plot_barlow_loss is not None:
    #         self.plot_loss(loss_li, loss_path=plot_barlow_loss, bin_width=seq2seq_config.train_steps // 500, validation=validation_loss_li)
    #     return seq2seq_model

    def train_seq2seq(
        self,
        cat2vec_datagen: C2VDataGen,
        seq2seq_config: Config,
        seq2seq_model: torch.nn.Module = None,
        plot_loss: Path = None,
        plot_barlow_loss: Path = None,
    ):
        def get_validation_loss(model, dg_validation: S2SDataGen):
            with torch.no_grad():
                example_loss_li = []
                Xs = []
                # for X in dg_validation._exhaustive_iter():
                for i, (X, w, mask) in zip(range(100), dg_validation):
                    if not PRELOAD_DATA_ON_GPU:
                        X = [{k: v.to(DEVICE) for k, v in e.items()} for e in X]
                        w = w.to(DEVICE)
                        mask = mask.to(DEVICE)
        
                    out_dict = model(X, w, sequence_mask=mask)
                    loss = out_dict['loss']

                    # out_dict_1 = model(X)
                    # out_dict_2 = model(X)
                    # encodings_1, decodings_1 = out_dict_1['encodings'], out_dict_1['decodings']
                    # encodings_2, decodings_2 = out_dict_2['encodings'], out_dict_2['decodings']
                    # loss = model.unsupervised_cse_loss(decodings_1, decodings_2)
                    
                    loss = loss.to(CPU).detach().numpy()
                    example_loss_li.append(loss)
            return np.mean(example_loss_li)

        dataset_device = CPU
        dataloader_num_workers = 3
        dataloader_pin_memory = True
        if PRELOAD_DATA_ON_GPU:
            dataset_device = GPU
            dataloader_num_workers = 0
            dataloader_pin_memory = False

        dg_train = S2SDataGen(
            df=self.data['train']['data'],
            cols=seq2seq_config.cols,
            inherit_encoding_lut=cat2vec_datagen._one_hot_encoding_lut,
            L=seq2seq_config.l,
            M=seq2seq_config.m,
            batch_size=1,#seq2seq_config.batch_size,
            p_mask_thresh=seq2seq_config.p_mask_thresh,
            device=dataset_device,
        )
        dg_validation = S2SDataGen(
            df=self.data['test']['data'],
            cols=seq2seq_config.cols,
            inherit_encoding_lut=cat2vec_datagen._one_hot_encoding_lut,
            L=seq2seq_config.l,
            M=seq2seq_config.m,
            device=dataset_device,
            batch_size=1,
            p_mask_thresh=seq2seq_config.p_mask_thresh,  # Don't randomly mask out input tokens
        )
        # dg_train._one_hot_lut = one_hot_lut
        # dg_validation._one_hot_lut = one_hot_lut
        dl_train = torch.utils.data.DataLoader(
            dg_train,
            batch_size=seq2seq_config.batch_size,
            num_workers=dataloader_num_workers,
            pin_memory=dataloader_pin_memory,

            # If whole dataset can fit onto GPU, do this:
            # num_workers=0,
            # If it can't, do this, and load the data on the CPU (in S2SDataGen)
            # num_workers=3,
            # pin_memory=True,
        )
        dl_validation = torch.utils.data.DataLoader(
            dg_validation,
            batch_size=seq2seq_config.batch_size,
            num_workers=dataloader_num_workers,
            pin_memory=dataloader_pin_memory,

            # If whole dataset can fit onto GPU, do this:
            # num_workers=0,
            # If it can't, do this, and load the data on the CPU (in S2SDataGen)
            # num_workers=3,
            # pin_memory=True,
        )
        if seq2seq_model is None:
            seq2seq_model = self._build_seq2seq_model(seq2seq_config, cat2vec_datagen.i_shape())
        # The next line enables weighted loss
        # seq2seq_model.weight_lut = dg.get_weight_lut()

        # AE TRAINING #

        optim = torch.optim.AdamW(
            seq2seq_model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
        )
        # optim = torch.optim.AdamW(
        #     [
        #         {'params': seq2seq_model.vectoriser.parameters(), 'weight_decay': 0.1, 'lr': 5e-5},
        #         {'params': seq2seq_model.custom_encoder.parameters(), 'weight_decay': 0.01},
        #         {'params': seq2seq_model.custom_decoder.parameters(), 'weight_decay': 0.01},
        #     ],
        #     lr=0.0001, 
        #     betas=(0.9, 0.98), 
        #     eps=1e-9
        # )
        # criterion = torch.nn.MSELoss(reduction='sum')
        # criterion = torch.nn.CosineSimilarity(dim=-1, eps=1e-08)
        losses = {
            "train": [],
            # "closeness": [],
            # "separation": [],
            # "consistency": [],
            # "mse": [],
            "validation": [],
        }
        loss_components = {}
        
        seq2seq_model.to(DEVICE)
        
        for t, (X, w, mask) in progress_with_status(
            zip(range(int(seq2seq_config.train_steps)), dl_train), 
            total=seq2seq_config.train_steps
        ):
            # print(X.shape, y.shape)
            # X = next(generator)
            # X = X.to(DEVICE)
            # y = {k: v.to(DEVICE) for k, v in y.items()}  # columnwise onehot labels
            # X = [{k: v.to(DEVICE) for k, v in e.items()} for e in X]
            # Xc = [{k: v.to(DEVICE) for k, v in e.items()} for e in Xc]
            # m = m.to(DEVICE)
            # w = {k: v.to(DEVICE) for k, v in w.items()}  # columnwise code weights for loss fn
            # X = dg_train._collate_fn(*X)
            # m = m.transpose(1, 0)
            # out_dict = seq2seq_model(X, Xc, sequence_mask=m, t=t)
            if not PRELOAD_DATA_ON_GPU:
                X = [{k: v.to(DEVICE) for k, v in e.items()} for e in X]
                w = w.to(DEVICE)
                mask = mask.to(DEVICE)

            out_dict = seq2seq_model(X, w, sequence_mask=mask, t=t)
            loss = out_dict['loss']

            # out_dict_1 = seq2seq_model(X, sequence_mask=m)
            # out_dict_2 = seq2seq_model(X, sequence_mask=m)
            # encodings_1, decodings_1 = out_dict_1['encodings'], out_dict_1['decodings']
            # encodings_2, decodings_2 = out_dict_2['encodings'], out_dict_2['decodings']
            # loss = seq2seq_model.unsupervised_cse_loss(decodings_1, decodings_2)

            # Calculate loss as mse between input and output
            # loss = criterion(X, X_hat)
            # Zero gradients, perform a backward pass, and update the weights.
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            # Report on the loss
            losses["train"].append(loss.to(CPU).detach().numpy())

            for name, value in out_dict['loss_components'].items():
                _value = value.to(CPU).detach().numpy()
                if name not in loss_components:
                    loss_components[name] = [_value]
                loss_components[name].append(_value)

            # losses["closeness"].append(np.mean(out_dict['closeness'].to(CPU).detach().numpy()))
            # losses["separation"].append(np.mean(out_dict['separation'].to(CPU).detach().numpy()))
            # losses["consistency"].append(np.mean(out_dict['consistency'].to(CPU).detach().numpy()))
            # losses["mse"].append(np.mean(out_dict['mse'].to(CPU).detach().numpy()))
            if t % (seq2seq_config.train_steps // seq2seq_config.validation_steps) == 0:
                losses["validation"].append(get_validation_loss(seq2seq_model, dl_validation))
                # validation_loss_li.append(get_validation_loss(seq2seq_model, dg_validation))
            if t % 10 == 0:
                update_status(
                    # f"Train={np.mean(losses["train"][-100:]):.3E}, " +
                    ", ".join([f"{name[:3]}={np.mean(li[-100:]):.2E}" for name, li in losses.items()]) + ". " +
                    ", ".join([f"{name[:3]}={np.mean(li[-100:]):.2E}" for name, li in loss_components.items()]) +
                    # ('' if len(validation_loss_li) == 0 else f", Validation={validation_loss_li[-1]:.3E}") + 
                    "."
                )
        losses = {k: np.asarray(v) for k, v in losses.items()}
        loss_components = {k: np.asarray(v) for k, v in loss_components.items()}
        STATUS_HANDLER = None
        if plot_loss is not None:
            self.plot_loss(
                losses, 
                loss_components,
                loss_path=plot_loss, 
            )

        # seq2seq_model = self._barlow_twins_tuning(
        #     seq2seq_config=seq2seq_config,
        #     seq2seq_model=seq2seq_model,
        #     dg_train=dg_train,
        #     dg_validation=dg_validation,
        #     plot_barlow_loss=plot_barlow_loss,
        # )
        np.savez_compressed(
            self.data_path.parent / "loss_data.npz",
            **losses
        )
        seq2seq_model.eval()

        return seq2seq_model

    @staticmethod
    def plot_loss(
        losses: Dict[str, np.ndarray], 
        loss_components: Dict[str, np.ndarray],
        loss_path: Path = None
    ):
        def get_mean_lb_ub(arr, w_len: int = 500):
            from numpy.lib.stride_tricks import sliding_window_view
            v_len = arr.shape[0]
            tmp_li = np.full(v_len + w_len - 1, np.nan)
            tmp_li[w_len//2:-w_len//2+1] = arr
            windows = sliding_window_view(tmp_li, w_len)
            mean = np.nanmean(windows, axis=-1)
            # ub = np.nanquantile(windows, 0.8, axis=-1)
            # lb = np.nanquantile(windows, 0.2, axis=-1)
            ub = mean + np.nanstd(windows, axis=-1)
            lb = mean - np.nanstd(windows, axis=-1)
            xs = list(range(arr.shape[0]))
            return xs, mean, ub, lb

        sns.set_theme(style="darkgrid")
        f, ax = plt.subplots(figsize=(7, 7))
        xs, mean, lb, ub = get_mean_lb_ub(losses['train'])
        plt.plot(xs, mean, label='train')
        xv = np.linspace(0, np.asarray(xs).max(), losses['validation'].shape[0])
        plt.plot(xv, losses['validation'], label='validation')
        for name, values in loss_components.items():
            xs, mean, lb, ub = get_mean_lb_ub(values)
            plt.plot(xs, mean, label=name, alpha=0.5)
        plt.legend()
        plt.xlabel("Train steps")
        plt.ylabel("Loss")
        ax.set_yscale('log')
        if loss_path is not None:
            plt.savefig(loss_path)
        plt.clf()
        matplotlib.rc_file_defaults()
        return 

    @staticmethod
    def _plot_embeddings(
        umap_embeddings: np.ndarray,
        dataset_targets: pd.Series,
        title: str,
        cluster_centres: np.ndarray = None,
        plot_save_path: Path = None,
        size: Tuple[int, int] = (12, 10),
        dpi: int = 180,
    ):
        def sort_fn(class_label: str):
            if isinstance(class_label, int):
                return class_label
            if class_label == "Unspecified":
                return -1
            if class_label.split('_')[-1].isnumeric():
                return int(class_label.split('_')[-1])
            return -1

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
        # Plot
        plt.clf()
        fig, ax = plt.subplots(figsize=size, dpi=dpi)
        for _class in classes:
            if _class == "Unspecified" or _class == -1:
                plt.scatter(ys[_class], xs[_class], label=_class, alpha=0.01, s=5, color='black')
            else:
                plt.scatter(ys[_class], xs[_class], label=_class, alpha=1, s=5)
        legend = plt.legend()
        for lh in legend.legendHandles:
            lh.set_alpha(1)
        if cluster_centres is not None:
            plt.scatter(
                cluster_centres[:, 0],
                cluster_centres[:, 1],
                marker="x",
                s=200,
                linewidths=3,
                color="black",
                # edgecolors="black",
                zorder=10,
            )
        plt.gca().set_aspect('equal', 'datalim')
        plt.title(title, fontsize=18)
        print("PLOTTING", plot_save_path)
        if plot_save_path is not None:
            plt.savefig(plot_save_path)
        plt.show()
        plt.clf()

    def _embed(self, features: np.ndarray, umap_kwargs: Dict = None, embed_ss=None):
        logging.info(f"Embedding to 2-dimensions with UMAP.")
        default_umap_kwargs = dict(
            n_neighbors=30,  # Choose UMAP parameters
            min_dist=0.0,    # which are better suited
            n_components=2,  # for clustering
            verbose=True,
            random_state=42,
            init='random'
        )
        kwargs = default_umap_kwargs
        if umap_kwargs is not None:
            kwargs.update(umap_kwargs)

        # features = features[:embed_ss, :]
        # features = StandardScaler(with_mean=True, with_std=False).fit_transform(features[:embed_ss, :])
        return umap.UMAP(**kwargs).fit_transform(features)

    def encode_cat2vec(
            self,
            cat2vec_model: torch.nn.Module,
            cat2vec_config: Config,
            cat2vec_datagen: C2VDataGen,
            dataset: str = 'full',
            plot: bool = True,
            plot_save_path: Path = None,
    ):
        dg = cat2vec_datagen
        logging.info(f"Getting cat2vec encodings for examples in dataset: {dataset}")
        cat2vec_encodings = [
            cat2vec_model.encode({
                col: dg._encoding_lut[col][example[col]]
                for col in cat2vec_config.cols
            }).detach().numpy()
            for _, example in tqdm(self.data[dataset]['data'].iterrows())
        ]
        encoding_df = pd.DataFrame(
            np.asarray(cat2vec_encodings).reshape(-1, cat2vec_config.n_state_features)
        )
        logging.info(f"Scaling the cat2vec encodings by mean & variance.")
        encoding_data = encoding_df[encoding_df.columns].values
        scaled_encoding_data = StandardScaler().fit_transform(encoding_data)
        if plot:
            umap_embeddings = self._embed(scaled_encoding_data)
            self._plot_embeddings(
                umap_embeddings=umap_embeddings,
                dataset_targets=self.data[dataset]['target'].iloc[:umap_embeddings.shape[0]],
                title=f"UMAP projection of the cat2vec encodings on set={dataset}",
                plot_save_path=plot_save_path,
            )
        return {
            'encodings': encoding_data,
            'embeddings': umap_embeddings if plot else None,
            'targets': self.data[dataset]['target'],
        }

    def encode_seq2seq(
            self,
            cat2vec_datagen: C2VDataGen,
            seq2seq_model: torch.nn.Module,
            seq2seq_config: Seq2SeqConfig,
            umap_kwargs: Dict = None,
            on: str = 'decodings',
            dataset: str = 'test',
            mode: str = 'last',
            noise=None,
            plot=True,
            join=True,
            embed_ss=None,
            plot_save_path: Path = None,
    ):

        dataset_device = CPU
        if PRELOAD_DATA_ON_GPU:
            dataset_device = GPU

        seq2seq_model.to(DEVICE)

        dg = S2SDataGen(
            df=self.data[dataset]['data'],
            cols=seq2seq_config.cols,
            inherit_encoding_lut=cat2vec_datagen._one_hot_encoding_lut,
            L=None,
            M=seq2seq_config.m,
            device=dataset_device,
            batch_size=1,#seq2seq_config.batch_size,
            p_mask_thresh=seq2seq_config.p_mask_thresh,
        )
        pw_event_encodings = {}
        for pid, (X, w, mask) in tqdm(zip(dg._patient_list, dg._exhaustive_iter()), total=len(dg._patient_list)):
            if not PRELOAD_DATA_ON_GPU:
                X = [{k: v.to(DEVICE) for k, v in e.items()} for e in X]
                w = w.to(DEVICE)
            out_dict = seq2seq_model(X, sequence_mask=mask)
            result = out_dict[on]
            pw_event_encodings[pid] = result.detach().cpu().numpy()
        encodings = np.concatenate(list(pw_event_encodings.values()), axis=0).squeeze()

        # # (2) Get Seq2Seq Encodings:
        # # (2a) Separate each patient sequence from the large pool of events in self.data[dataset]['data']
        patient_ids = self.data[dataset]['data']['patient_id'].unique().tolist()
        patient_dfs = {
            patient_id: self.data[dataset]['data'][self.data[dataset]['data'].patient_id == patient_id]
            for patient_id in patient_ids
        }
        # # (2b) For each patient sequence, produce a sequence of Cat2Vec encodings
        # pw_event_encodings = {
        #     patient_id: [
        #         cat2vec_model.encode({
        #             col: c2v_dg._encoding_lut[col][example[col]]
        #             for col in cat2vec_config.cols
        #         }).detach().numpy()
        #         for _, example in patient_df.iterrows()
        #     ]
        #     for patient_id, patient_df in tqdm(patient_dfs.items())
        # }
        # # (2c) Get the corresponding targets for each event in each patient sequence
        pw_targets = {
            patient_id: self.data[dataset]['target'][patient_df.index]
            for patient_id, patient_df in patient_dfs.items()
        }
        # # (2d) For each event E in each patient sequence, use history
        # # of at most m events to produce a seq2seq encoding for the
        # # sequence. The subsequent history-aware encoding for event E
        # # is the final encoding in the seq2seq encoded
        # # sequence. Optionally, this encoding can be concatenated with
        # # the cat2vec encoding for that event.
        # def get_seq2seq_encodings(seq):
        #     # return seq2seq_model._encode(seq)
        #     mask = seq2seq_model.get_mask(seq.shape[0]).to(seq.device)
        #     return seq2seq_model.tr.encoder(seq, mask=mask)

        # m = seq2seq_config.m
        # encodings = []
        # for patient_id, vec_encodings in tqdm(pw_event_encodings.items()):
        #     if mode == 'all':
        #         vec_encodings = np.asarray(vec_encodings)
        #         l = vec_encodings.shape[0]
        #         seq = torch.Tensor(vec_encodings).reshape((l, 1, -1))
        #         seq_encodings = get_seq2seq_encodings(seq).detach().numpy().reshape((l, -1))
        #         if join:
        #             enc = np.concatenate((vec_encodings, seq_encodings), axis=-1)
        #             for idx in range(enc.shape[0]):
        #                 encodings.append(enc[idx, :])
        #         else:
        #             for idx in range(seq_encodings.shape[0]):
        #                 encodings.append(seq_encodings[idx, :])
        #     elif mode == 'last':
        #         for i in range(len(vec_encodings)):
        #             start = max(0, i - seq2seq_config.l + 1)
        #             end = i + 1
        #             l = end - start
        #             seq = torch.Tensor(np.asarray(vec_encodings[start:end])).reshape((l, 1, m))
        #             seq_encodings = get_seq2seq_encodings(seq).detach().numpy().reshape((l, m))
        #             if join:
        #                 encodings.append(np.concatenate((vec_encodings[i], seq_encodings[-1, ...])))
        #             else:
        #                 encodings.append(seq_encodings[-1, ...])

        # Optionally, add noise to the encodings
        noisy_encodings = []
        if noise is not None:
            for encoding in encodings:
                noisy_encodings.append(encoding + np.random.normal(0., noise, (encoding.shape[0])))
            encodings = noisy_encodings

        targets = pd.Series([t for patient_targets in pw_targets.values() for t in patient_targets])
        if plot:
            # Perform a UMAP embedding of the encodings
            umap_embeddings = self._embed(np.asarray(encodings), umap_kwargs, embed_ss=embed_ss)

            pw_seq_embeddings = {}
            start_idx = 0
            for patient_id, vec_encodings in tqdm(pw_event_encodings.items()):
                end_idx = start_idx + np.asarray(vec_encodings).shape[0]
                pw_seq_embeddings[patient_id] = {
                    'embeddings': umap_embeddings[start_idx:end_idx],
                    'targets': targets[start_idx:end_idx],
                }
                start_idx = end_idx

            self._plot_embeddings(
                umap_embeddings=umap_embeddings,
                dataset_targets=targets.iloc[:umap_embeddings.shape[0]],
                title=(
                    f"UMAP projection of the {'cat2vec|' if join else ''}seq2seq encodings on set={dataset}\n"
                    f"Labels: Truth"
                ),
                plot_save_path=plot_save_path,
            )
            assert len(targets) == np.asarray(encodings).shape[0], f"{len(targets)=} != {np.asarray(encodings).shape[0]=}"
        return {
            'encodings': np.asarray(encodings),
            'targets': targets,
            'embeddings': umap_embeddings if plot else None,
            'pw_seq_embeddings': pw_seq_embeddings if plot else None,
        }

    @staticmethod
    def cluster(
            encodings: np.ndarray,
            targets: pd.Series = None,
            cluster_method: str = 'kmeans',
            cluster_kwargs: Dict = None,
            classify_noise: bool = True,
    ):
        # if targets is None:
        #     targets = self.data[dataset]['target']
        targets = pd.Series(targets, name='target')
        if len(targets) != encodings.shape[0]:
            raise ValueError(
                "The embeddings and targets lengths are unequal. "
                "Check the dataset which was embedded with Experiment.embed()."
            )
        print(f"Performing clustering using method {cluster_method}.")
        cluster_kwargs = {} if cluster_kwargs is None else cluster_kwargs
        # Perform clustering
        if cluster_method == 'kmeans':
            # kmeans
            clusters = KMeans(**cluster_kwargs, random_state=42).fit(encodings).labels_
        elif cluster_method == 'spectral':
            # spectral
            clusters = SpectralClustering(
                **cluster_kwargs, random_state=42, assign_labels='discretize'
            ).fit(encodings).labels_
        elif cluster_method == 'HDBSCAN':
            # hdbscan
            clusters = HDBSCAN(**cluster_kwargs).fit_predict(encodings)
        else:
            raise NotImplementedError()
        
        # Give cluster classes distinct names (not simple integers)
        est_targets = pd.Series(clusters, name='est_target').apply(
            lambda x: "Unspecified" if x == -1 else f"cluster_{x}"
        )
        if classify_noise and cluster_method == 'HDBSCAN':
            # If unspecified
            unspec_indices = est_targets[est_targets == "Unspecified"].index.tolist()
            spec_indices = est_targets[est_targets != "Unspecified"].index.tolist()
            spec_encodings = encodings[np.asarray(spec_indices), :]
            spec_targets_ri = est_targets[spec_indices].reset_index()
            choices = {}
            for unspec_index in tqdm(unspec_indices):
                val = encodings[unspec_index, :]
                mse = ((spec_encodings - val)**2).sum(axis=-1) / val.shape[0]
                idxs = np.argsort(mse)[1:21]
                candidates = spec_targets_ri[spec_targets_ri.index.isin(idxs)]
                choices[unspec_index] = candidates.est_target.value_counts().index[0]
            for unspec_index, choice in choices.items():
                est_targets.iloc[unspec_index] = choice

        classifications = pd.DataFrame([targets, est_targets]).transpose()
        return classifications

    @staticmethod
    def score(
            classifications: pd.DataFrame,
            plot_cm: bool = True,
            plot: bool = False,
            umap_embeddings: np.ndarray = None,
            plot_save_path: Path = None,
    ):
        targets, est_targets = classifications['target'], classifications['est_target']
        # Score
        ami = adjusted_mutual_info_score(targets, est_targets)
        ami_star = adjusted_mutual_info_score(
            targets[est_targets != 'Unspecified'], est_targets[est_targets != 'Unspecified']
        )
        # Confusion Matrix
        target_classes = est_targets.unique()
        target_classes.sort()
        target_classes = target_classes.tolist()
        targetwise_classifications = {
            target: classifications[classifications['target'] == target]
            for target in targets
        }
        # <Dict>[<str>'target', <Dict>[<str>'est_target', <int>count]]
        cm = {
            t:
                {
                    target_class:
                    targetwise_classifications[t].est_target.value_counts().to_dict().get(target_class, 0)
                    for target_class in target_classes
                }
            for t in targetwise_classifications.keys()
        }
        cm = dict(sorted(cm.items(), key=lambda item: np.argmax(list(item[1].values()))))
        cm_arr = np.asarray([list(x.values()) for x in cm.values()])

        if plot_cm:
            print("Plotting confusion matrix")
            fig, ax = plt.subplots()
            sns.heatmap(cm_arr, square=True, annot=True, fmt='g', ax=ax, cbar=False,cmap="viridis")
            ax.set_xlabel('Predicted labels')
            ax.set_ylabel('True labels')
            ax.set_title(f'Confusion Matrix\nAdjusted Mutual Information={(100*ami):.2f}% ({(100*ami_star):.2f}%*)')
            ax.set_xticks(list(range(len(target_classes))))
            ax.xaxis.set_ticklabels(target_classes, rotation=45)
            ax.set_yticks(list(range(len(cm.keys()))))
            ax.yaxis.set_ticklabels(list(cm.keys()), rotation='horizontal')
            plt.show()

        if plot:
            Experiment._plot_embeddings(
                umap_embeddings=umap_embeddings,
                dataset_targets=est_targets,
                title=(
                    f"UMAP projection of the encodings\nLabels: Predicted"
                ),
                plot_save_path=plot_save_path,
            )
        return cm_arr, targets, est_targets, ami, ami_star
