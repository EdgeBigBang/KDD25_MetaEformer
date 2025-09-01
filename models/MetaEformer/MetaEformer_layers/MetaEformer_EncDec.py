import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.MetaEformer.utils import get_activation_fn


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        bias = True
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.ff(y)
        y = self.dropout(y)


        return self.norm2(x + y), attn


class EchoLayer(nn.Module):
    def __init__(self, d_model, pattern_num, pattern_len, enc_seq_len, device, low_layer, sim_num):
        super(EchoLayer, self).__init__()
        self.dim_val = d_model
        half = int(d_model/2)
        self.patch_seq_len = pattern_len
        self.enc_seq_len = enc_seq_len
        self.device = device
        self.mpp_size = pattern_num
        self.low_layer = low_layer
        self.sim_num = sim_num
        self.update_count = 0

        self.fuse_src = nn.Linear(
            in_features= self.patch_seq_len * half,
            out_features=self.sim_num
            )

        self.recover = nn.Linear(
            in_features=self.sim_num,
            out_features=half
        )

        self.low_layer_MPP = nn.Linear(
            in_features=half,
            out_features=10
        )

    def forward(self, src, meta_pattern_pool):

        half_dim = int(self.dim_val / 2)  # 1/2D
        src_half = src[:, :, half_dim:].flatten(1).clone()  # B * (T*1/2D)
        patches = src_half.reshape(-1, int(self.enc_seq_len/self.patch_seq_len), self.patch_seq_len, half_dim)

        patches_low_dim = self.low_layer_MPP(patches)  
        patches_low_dim_mean = torch.mean(patches_low_dim, dim=-1)  # [batch, num_patches, pattern_length]  

        output_features = torch.empty((src.shape[0], 0, half_dim)).to(self.device)
        padding_features = torch.empty((src.shape[0], self.enc_seq_len)).to(self.device)

        for patch_idx in range(patches.shape[1]):
            current_patch = patches[:, patch_idx, :, :]  
            
            current_patch_low = patches_low_dim_mean[:, patch_idx, :]  
            
            similarity_matrix = current_patch_low.unsqueeze(1) * meta_pattern_pool.unsqueeze(0)
            similarity_scores = similarity_matrix.sum(dim=2)
            topk_scores, topk_indices = torch.topk(similarity_scores, self.sim_num, dim=1)

            selected_patterns = meta_pattern_pool[topk_indices]  

            fusion_input = current_patch.reshape(-1, self.patch_seq_len * half_dim)
            fusion_weights_raw = self.fuse_src(fusion_input)
            self.update_count += 1
            fusion_weights = torch.softmax(fusion_weights_raw, dim=1)
     

            patterns_for_fusion = selected_patterns.transpose(-2, -1).to(self.device)  # [batch, pattern_length, sim_num]
           
            fused_features = torch.einsum('bk,bpk->bpk', fusion_weights, patterns_for_fusion)
            
            padding_features_current = torch.einsum('bk,bkp->bp', fusion_weights, selected_patterns)
   
            padding_features = torch.cat((padding_features, padding_features_current), dim=-1)

            recovered_features = self.recover(fused_features)
            output_features = torch.cat((output_features, recovered_features), dim=1)

        final_output = torch.cat((src[:, :, :half_dim], output_features), dim=-1)  

        return final_output, padding_features


class MPPBuilder(nn.Module):
    def __init__(self, low_layer):
        super(MPPBuilder, self).__init__()
        self.low_layer = low_layer

    def forward(self, enc_out, mpp, decomp):
        low_enc = self.low_layer(enc_out)  # B*T*D  -> B*T*d
        ser_re = torch.mean(low_enc, dim=-1)  # B*T

        season, _ =  decomp(ser_re)
        season_no_grad = season.detach()

        if not mpp.s_init:
            mpp = mpp.build_pool_seasonal(season_no_grad)
        else:
            mpp = mpp.update_pool(season_no_grad)

        return mpp

class EchoEncoder(nn.Module):
    def __init__(self, attn_layers, mpp_builders, echo_layers, conv_layers=None, norm_layer=None):
        super(EchoEncoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.mpp_builders = nn.ModuleList(mpp_builders)
        self.echo_layers = nn.ModuleList(echo_layers)
        self.norm = norm_layer

    def forward(self, x, MPP, stl, MPP_update_flag, attn_mask=None):
        # x [B, L, D]
        mpp = []
        attns = []

        for attn_layer, mpp_builder, echo_layer in zip(self.attn_layers, self.mpp_builders, self.echo_layers):
            x, attn = attn_layer(x, attn_mask=attn_mask)
            if MPP_update_flag:
                mpp = mpp_builder(x, MPP, stl)
            if MPP != None:
                x, padding = echo_layer(x, MPP.seasonal_pool)

            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        if MPP != None:
            return x, padding, attns, mpp
        else:
            return x, None, attns, mpp


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x


