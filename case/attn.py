"""
Derived from https://jaketae.github.io/study/relative-positional-encoding/
"""


import copy
import math
from typing import Optional, Any, Union, Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_
from torch.nn import (
    Module, 
    ModuleList,
    Dropout,
    Linear,
    LayerNorm,
)


MAX_LEN = 1024 * 16


class RelativeGlobalAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_len=MAX_LEN, dropout=0.1):
        super().__init__()
        d_head, remainder = divmod(d_model, num_heads)
        if remainder:
            raise ValueError(
                "incompatible `d_model` and `num_heads`"
            )
        self.max_len = max_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.Er = nn.Parameter(torch.randn(max_len, d_head))
        # self.register_buffer(
        #     "mask", 
        #     torch.tril(torch.ones(max_len, max_len))
        #     .unsqueeze(0).unsqueeze(0)
        # )
        # self.mask.shape = (1, 1, max_len, max_len)

    def forward(self, q, k, v, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True):
        # x.shape == (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = q.shape
        
        if seq_len > self.max_len:
            raise ValueError(
                "sequence length exceeds model capacity"
            )

        # merge key padding and attention masks
        # This code is derived from torch.nn.Functional.multi_headed_attention_forward
        if key_padding_mask is not None:
            if key_padding_mask.shape[0] != batch_size:
                key_padding_mask = key_padding_mask.transpose(1, 0)
            assert key_padding_mask.shape == (batch_size, seq_len), \
                f"expecting key_padding_mask shape of {(batch_size, seq_len)}, but got {key_padding_mask.shape}"
            key_padding_mask = key_padding_mask.view(batch_size, 1, 1, seq_len).   \
                expand(-1, self.num_heads, -1, -1).reshape(batch_size * self.num_heads, 1, seq_len)
            if attn_mask is None:
                attn_mask = key_padding_mask
            elif attn_mask.dtype == torch.bool:
                key_padding_mask = key_padding_mask.to(attn_mask.device)
                attn_mask = attn_mask.logical_or(key_padding_mask)
            else:
                key_padding_mask = key_padding_mask.to(attn_mask.device)
                attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))
            attn_mask = attn_mask.reshape(batch_size, self.num_heads, seq_len, seq_len)

        k_t = self.key(k).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        # k_t.shape = (batch_size, num_heads, d_head, seq_len)
        v = self.value(v).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(q).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # shape = (batch_size, num_heads, seq_len, d_head)
        
        start = self.max_len - seq_len
        Er_t = self.Er[start:, :].transpose(0, 1)
        # Er_t.shape = (d_head, seq_len)
        QEr = torch.matmul(q, Er_t)
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        Srel = self.skew(QEr)
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
        QK_t = torch.matmul(q, k_t)
        # QK_t.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = (QK_t + Srel) / math.sqrt(q.size(-1))
        # mask = self.mask[:, :, :seq_len, :seq_len]
        # mask.shape = (1, 1, seq_len, seq_len)
        # attn = attn.masked_fill(mask == 0, float("-inf"))
        if attn_mask is not None:
            attn += attn_mask
            assert not torch.isnan(attn).any(), "attn contains nan"
        # attn.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        # out.shape = (batch_size, num_heads, seq_len, d_head)
        out = out.transpose(1, 2)
        # out.shape == (batch_size, seq_len, num_heads, d_head)
        out = out.reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)
        return self.dropout(out)
    
    def skew(self, QEr):
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        padded = F.pad(QEr, (1, 0))
        # padded.shape = (batch_size, num_heads, seq_len, 1 + seq_len)
        batch_size, num_heads, num_rows, num_cols = padded.shape
        reshaped = padded.reshape(batch_size, num_heads, num_cols, num_rows)
        # reshaped.size = (batch_size, num_heads, 1 + seq_len, seq_len)
        Srel = reshaped[:, :, 1:, :]
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
        return Srel


class TransformerEncoderRPALayer(Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderRPALayer, self).__init__()
        self.self_attn = RelativeGlobalAttention(
            d_model=d_model,
            num_heads=nhead,
            max_len=MAX_LEN,
            dropout=dropout,
        )
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.batch_first = batch_first

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderRPALayer, self).__setstate__(state)

    def forward(
        self, 
        src: Tensor, 
        src_mask: Optional[Tensor] = None, 
        src_key_padding_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        x = src
        if not self.batch_first:
            x = x.permute(1, 0, 2)
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
        if not self.batch_first:
            x = x.permute(1, 0, 2)
        return x

    # self-attention block
    def _sa_block(
        self, 
        x: Tensor,
        attn_mask: Optional[Tensor], 
        key_padding_mask: Optional[Tensor]
    ) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransformerDecoderRPALayer(Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerDecoderRPALayer, self).__init__()
        self.self_attn = RelativeGlobalAttention(
            d_model=d_model,
            num_heads=nhead,
            max_len=MAX_LEN,
            dropout=dropout,
        )
        self.multihead_attn = RelativeGlobalAttention(
            d_model=d_model,
            num_heads=nhead,
            max_len=MAX_LEN,
            dropout=dropout,
        )
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        self.batch_first = batch_first

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderRPALayer, self).__setstate__(state)

    def forward(
        self, 
        tgt: Tensor, 
        memory: Tensor, 
        tgt_mask: Optional[Tensor] = None, 
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None, 
        memory_key_padding_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        x = tgt
        if not self.batch_first:
            x = x.permute(1, 0, 2)

        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
            x = self.norm3(x + self._ff_block(x))

        if not self.batch_first:
            x = x.permute(1, 0, 2)

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
