from typing import Dict, List
import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from .attn import TransformerEncoderRPALayer, TransformerDecoderRPALayer


class Vectoriser(nn.Module):
    def __init__(
        self,
        column_cardinality: Dict,
        n_state_features: int,
        embedding_dim: int,
        dropout: float,
    ):
        super(Vectoriser, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.cols = list(column_cardinality.keys())
        self.column_cardinality = column_cardinality
        self.embedding_dim = embedding_dim
        # Vectorise codes
        # self.col_features = nn.ModuleDict({
        #     col: torch.nn.Linear(col_len, self.embedding_dim)
        #     for col, col_len in column_cardinality.items()
        # })
        # self.state_features = torch.nn.Linear(
        #     len(column_cardinality)*self.embedding_dim, n_state_features
        # )
        self.state_features = torch.nn.Linear(
            sum(column_cardinality.values()), n_state_features
        )

    def _vectorise_codes(self, codes: Dict[str, Tensor]):
        if len(list(codes.values())[0].shape) == 1:
            codes = {k: v.unsqueeze(0) for k, v in codes.items()}

        _join_cols = torch.cat(list(codes.values()), axis=-1)
        state_features = self.state_features(_join_cols)
        assert not torch.isnan(state_features).any(), f"state_features contains nan\n{state_features.shape}"
        return state_features

    def forward(self, codes: Dict[str, Tensor]) -> Tensor:
        vectorised_events = [self._vectorise_codes(raw_events) for raw_events in codes]
        vectorised_events = torch.stack(vectorised_events)
        return vectorised_events


class Seq2Seq(nn.Module):
    def __init__(
        self,
        column_cardinality: Dict,
        n_state_features: int,
        num_heads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        feedforward_dim: int,
        dropout: float,
        encoder_window_size: int = 2,
        attention_window_len: int = 4,
        use_embedding_loss: bool = True,
        weight_lut: Dict[str, Tensor] = None,
        temp: float = 0.05,
    ):
        super(Seq2Seq, self).__init__()
        self.cols = list(column_cardinality.keys())
        self.column_cardinality = column_cardinality
        self.embedding_dim = n_state_features
        self.weight_lut = weight_lut
        # {
        #     columnwise_code_weights[col].reshape(L*B) if columnwise_code_weights is not None else None
        #     for col in self.cols
        # }
        self.vectoriser = Vectoriser(
            column_cardinality, n_state_features, n_state_features, dropout=dropout
        )

        # # Relative encoding
        self.pe = None
        encoder_layer = TransformerEncoderRPALayer(
            d_model=self.embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
        )
        decoder_layer = TransformerDecoderRPALayer(
            d_model=self.embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
        )

        # # Absolute encoding
        # self.pe = PositionalEncoding(self.embedding_dim, dropout=dropout)
        # encoder_layer = torch.nn.TransformerEncoderLayer(
        #     d_model=self.embedding_dim,
        #     nhead=num_heads,
        #     dim_feedforward=feedforward_dim,
        #     dropout=dropout,
        # )
        # decoder_layer = torch.nn.TransformerDecoderLayer(
        #     d_model=self.embedding_dim,
        #     nhead=num_heads,
        #     dim_feedforward=feedforward_dim,
        #     dropout=dropout,
        # )

        # # No positional encoding
        # self.pe = None
        # encoder_layer = torch.nn.TransformerEncoderLayer(
        #     d_model=self.embedding_dim,
        #     nhead=num_heads,
        #     dim_feedforward=feedforward_dim,
        #     dropout=dropout,
        # )
        # decoder_layer = torch.nn.TransformerDecoderLayer(
        #     d_model=self.embedding_dim,
        #     nhead=num_heads,
        #     dim_feedforward=feedforward_dim,
        #     dropout=dropout,
        # )

        self.custom_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers,
        )
        self.custom_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_decoder_layers,
        )
        # self.tr = nn.Transformer(
        #     d_model=self.embedding_dim,
        #     nhead=num_heads,
        #     # num_encoder_layers=num_encoder_layers,
        #     # num_decoder_layers=num_decoder_layers,
        #     dim_feedforward=feedforward_dim,
        #     dropout=dropout,
        #     custom_encoder=self.custom_encoder,
        #     custom_decoder=self.custom_decoder,
        # )
        # assert self.embedding_dim**0.5 % 1 == 0
        # self.downsampler = nn.Linear(self.embedding_dim, int(self.embedding_dim**0.5))
        # self.upsampler = nn.Linear(int(self.embedding_dim**0.5), self.embedding_dim)
        # self.col_fc_out = nn.ModuleDict({
        #     col_name: nn.Linear(self.embedding_dim, col_cardinality)
        #     for col_name, col_cardinality in self.column_cardinality.items()
        # })
        # self.fc_out = nn.Linear(self.embedding_dim, self.column_cardinality)
        self.mse_loss = nn.MSELoss(reduce=True, reduction="mean")
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduce=True, reduction="mean")
        self.dropout = nn.Dropout(dropout)
        self.attention_window_len = attention_window_len
        self.use_embedding_loss = use_embedding_loss
        self.device = None
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)
        self.dist = torch.nn.PairwiseDistance()
        self.masks = {}
        self.kernel = None
        self.encoder_window_size = encoder_window_size
        # self.l2norm = torch.linalg.norm()

    def get_tril_mask(self, n: int):
        mask = torch.tril(torch.ones(n, n))
        mask = mask.masked_fill(mask == 0, float("-inf"))
        mask = mask.masked_fill(mask == 1, float(0))
        return mask

    def get_mask(self, n: int):
        """This method returns a mask where the position i in n is masked out.
        This means that the model will learn to use positions j in the sequence
        to predict position i, where i != j.

        i.e., If the sequence length n is 3, then the mask will look like:

        [[-inf,    0,    0],
         [   0, -inf,    0],
         [   0,    0, -inf]]
        """
        return torch.diagflat(torch.Tensor([float("-inf") for _ in range(n)]))

    def get_windowed_mask(self, n: int, l: int):
        """If the sequence length n is 4, and l is 2, then the mask will look like:

        [[   0,    0,    0, -inf],
         [   0,    0,    0,    0],
         [   0,    0,    0,    0],
         [-inf,    0,    0,    0]]
        """
        if l + 1 >= n:
            # If the sequence is too short, just view set the window size to the sequence size.
            l = n - 1
        assert l < n
        if self.masks.get((n, l), False) is not False:
            return self.masks[(n, l)]
        tril = torch.tril(torch.full((n, n), float("-inf")), diagonal=-(l + 1))
        mask = tril + tril.T
        self.masks[(n, l)] = mask
        return mask

    def _encode(self, src: Tensor, tgt: Tensor, sequence_mask: Tensor = None) -> Tensor:
        attn_mask = self.get_windowed_mask(src.shape[0], self.encoder_window_size).to(device=src.device)
        # attn_mask = torch.zeros((src.shape[0], src.shape[0]), device=src.device)
        # attn_mask = self.get_windowed_mask(src.shape[0], 2).to(device=src.device)
        encodings = self.custom_encoder(src, mask=attn_mask, src_key_padding_mask=sequence_mask)

        # attn_mask = torch.zeros((src.shape[0], src.shape[0]), device=src.device)
        # attn_mask = torch.tril(torch.full((src.shape[0], src.shape[0]), float("-inf")), diagonal=-1).T.to(device=src.device)
        attn_mask = self.get_mask(src.shape[0]).to(device=src.device)

        decodings = self.custom_decoder(
            tgt,
            encodings,
            tgt_mask=attn_mask,
            memory_mask=attn_mask,
            memory_key_padding_mask=sequence_mask,
            tgt_key_padding_mask=sequence_mask,
        )

        return encodings, decodings

    def forward(
        self,
        # List.len = L, Tensor.shape = (B)
        onehot_event_sequence: List[Dict[str, Tensor]],
        weights: torch.Tensor = None,
        # List.len = L, Tensor.shape = (B)
        raw_event_sequence: List[Dict[str, Tensor]] = None,
        sequence_mask: Tensor = None,
        t=None,
    ):
        vectorised_events = self.vectoriser(onehot_event_sequence)
        L, B, M = vectorised_events.shape

        src = tgt = vectorised_events

        if self.pe is not None:
            src = self.pe(src)
            tgt = self.pe(tgt)

        if weights is None:
            weights = torch.ones((L, B))
        else:
            weights = weights.transpose(1, 0)
        Lw, Bw = weights.shape
        weights = weights.to(vectorised_events.device)

        assert L == Lw and B == Bw, "shapes of weights and sequence don't align."
        # Encode
        encodings, decodings = self._encode(src, tgt, sequence_mask=sequence_mask)

        # Semantic-Temporal Learning Objective
        encoding_loss_components = self.get_losses(encodings, weights, mask=sequence_mask, t=t)
        decoding_loss_components = self.get_losses(decodings, weights, mask=sequence_mask, t=t)
        loss = (
            decoding_loss_components["closeness"],
            decoding_loss_components["separation"],
            encoding_loss_components["consistency"],
        )
        loss_components = {
            "closeness": decoding_loss_components["closeness"],
            "separation": decoding_loss_components["separation"],
            'consistency': encoding_loss_components["consistency"],
            'mean': decoding_loss_components["mean"],
            'maximum': decoding_loss_components["maximum"],
        }

        # # custom + mse
        # closeness, separation, _ = self.get_losses(encodings, weights, t=t)
        # mse = self.mse_loss(tgt, decodings)
        # loss = closeness + separation + mse
        # loss_components = {
        #     "closeness": closeness,
        #     "separation": separation,
        #     "mse": mse,
        # }

        # # mse only
        # # mse_enc = self.mse_loss(tgt, encodings)
        # mse_dec = self.mse_loss(tgt, decodings)
        # loss = mse_dec
        # loss_components = {
        #     "mse": mse_dec,
        # }

        return {
            "vectorised_events": vectorised_events,
            "encodings": encodings,
            "decodings": decodings,
            "loss": loss,
            "loss_components": loss_components,
        }

    @staticmethod
    def adjacent_distances(T: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        L, B, M = T.shape
        if L < 2:
            return torch.full((L - 1, B), torch.nan)
        T1 = T[:-1, :, :]
        T2 = T[1:, :, :]
        distances = (T1 - T2) ** 2
        w = w.transpose(1, 0)
        w = F.max_pool1d(w, kernel_size=2, stride=1)
        w = w.transpose(1, 0)
        return distances.sum(dim=-1).sqrt() * w



    def stlo(self, e, w, g, reduce=True):
        """
        Batched Semantic-Temporal Learning Objective
        """
        L, B, _ = e.shape
        distances = self.adjacent_distances(e, w)
        if distances.shape[0] < 2:
            return 0, 0, 0
        distances_sorted = torch.sort(distances, dim=0)[0]

        n_state_changes = 1
        maximum = distances_sorted[-1*n_state_changes:, :].mean(dim=0)
        mean = distances_sorted[:-1*n_state_changes, :].mean(dim=0)
        std = distances_sorted[:-1*n_state_changes, :].std(dim=0)

        consistency = std
        closeness = (mean / torch.clamp(maximum, min=1e-8, max=g))
        separation = (1 - torch.tanh((maximum - mean) / (2 * g)))
        components = dict(
            consistency=consistency,
            closeness=closeness,
            separation=separation,
            mean=mean,
            maximum=maximum,
        )
        if reduce:
            components = {k: v.mean() for k, v in components.items()}
        return components

    def get_losses(
        self,
        embeddings: Tensor,
        weights: Tensor,
        n_state_changes: int = 1,
        separation_gain: float = 10,
        mask: Tensor = None,
        t=None,
    ) -> Tensor:
        if mask is not None and mask.any().item():
            # Until torch.masked has matured, we need to do this iteratively...
            L, B, _ = embeddings.shape

            # # Get loss piece-by-piece, naive and slow
            # running_components = dict()
            # for b in range(B):
            #     # Pretend it has a batch size of 1
            #     n_valid = (~mask[b:b+1, :]).sum()  # mask is active high
            #     e, w = embeddings[:n_valid, b:b+1, :], weights[:n_valid, b:b+1]
            #     components = self.stlo(e, w, separation_gain)
            #     for name, component in components.items():
            #         if name not in running_components:
            #             running_components[name] = torch.zeros(B, device=embeddings.device)
            #         running_components[name][b] = component

            # Run loss for all masks in batch on the entire batch:
            # More computation, but faster due to batching, especially on 
            n_valid_losses = {}
            min_n_valid = (~mask).sum(axis=1).min().item() # mask is active high
            for n_valid in range(min_n_valid, L + 1):
                e, w = embeddings[:n_valid, :, :], weights[:n_valid, :]
                n_valid_losses[n_valid] = self.stlo(e, w, separation_gain, reduce=False)
            # Now get the losses based on the actual masks
            running_components = {}
            for b in range(B):
                n_valid = (~mask[b:b+1, :]).sum().item()  # mask is active high
                components = n_valid_losses[n_valid]
                for name, component in components.items():
                    if name not in running_components:
                        running_components[name] = torch.zeros(B, device=embeddings.device)
                    running_components[name][b] = component[b]

            return {k: v.mean() for k, v in running_components.items()}

        # We can compute loss on whole batch at once if there is no sequence mask
        return self.stlo(embeddings, weights, separation_gain)


