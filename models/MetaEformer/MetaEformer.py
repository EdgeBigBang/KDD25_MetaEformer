import sys

import torch
import torch.nn as nn
import os
CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]
sys.path.append(CURRENT_DIR)

from MetaEformer_layers.MetaEformer_EncDec import Decoder, DecoderLayer, EchoEncoder, MPPBuilder, EncoderLayer, EchoLayer
from MetaEformer_layers.SelfAttention_Family import FullAttention, AttentionLayer
from MetaEformer_layers.Embed import DataEmbedding, StaticContextEmbedding
from MetaEformer_layers.series_decomp import series_decomp
from MetaEformer_layers.MetaPatternPool import MetaPatternPool as MPP


class MetaEformer(nn.Module):
    """
    MetaEformer
    """
    def __init__(self, configs):
        super(MetaEformer, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.device = configs.device

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # static
        self.static_layer = StaticContextEmbedding(configs.d_model, configs.d_model, configs.dim_static, configs.dropout)
        self.if_padding = configs.if_padding

        # MetaPatternpool
        self.MPP = MPP(configs.wave_class, configs.wave_len, configs.threshold, self.device)
        self.decomp = series_decomp(configs.kernel_size)
        self.low_layer = nn.Linear(
            in_features=configs.d_model,
            out_features=configs.low_dim
        )

        # Encoder
        self.echoencoder = EchoEncoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],

            [MPPBuilder(self.low_layer) for l in range(configs.e_layers)],
            [EchoLayer(configs.d_model, configs.wave_class, configs.wave_len, configs.enc_len, self.device, self.low_layer, configs.sim_num) for l in range(configs.e_layers)],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_static, MPP_update_flag,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        enc_out, Echo_padding, attns, mpp = self.echoencoder(enc_out, self.MPP, self.decomp, MPP_update_flag, attn_mask=enc_self_mask)

        if self.MPP != None:
            if self.if_padding:  # padding
                if Echo_padding.shape[-1] >= self.pred_len:

                    x_dec[:, -self.pred_len:, 0] = Echo_padding[:, -self.pred_len:]
                else:
                    repeat_factor = (self.pred_len // Echo_padding.shape[-1]) + 1
                    extended_global_padding = Echo_padding.repeat(1, repeat_factor)
                    x_dec[:, -self.pred_len:, 0] = extended_global_padding[:, -self.pred_len:]
            else:
                x_dec[:, -self.pred_len:, 0] = 0

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        if x_static != None:
            dec_out = self.static_layer(dec_out, x_static)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns, mpp
        else:
            return dec_out[:, -self.pred_len:, :], mpp  # [B, L, D]
