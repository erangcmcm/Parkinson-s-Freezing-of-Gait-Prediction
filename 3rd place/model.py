import json
import math
import random
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
try: from lion_pytorch import Lion
except: pass

from collections import defaultdict

import timm
import transformers
import segmentation_models_pytorch as smp

from timm.models.vision_transformer import VisionTransformer
from timm.models.vision_transformer_relpos import VisionTransformerRelPos
from timm.models.layers import PatchEmbed

from sklearn.metrics import average_precision_score
from types import SimpleNamespace
import inspect
import secrets

torch.manual_seed(datetime.datetime.now().microsecond)

# send only the relevant params to the model
def getParams(cls, params):
    p = {k: params[k]
        for k in set(inspect.getfullargspec(cls).args) 
                    & set(params.keys())}
    print(p)
    return p

# sinusioid positional encoding, for given sequence length and embedding size, and tunable parameter n
def getPositionalEncoding(seq_len, d_model, n = 10000):
    pos = torch.arange(0, seq_len).unsqueeze(1)
    i = torch.arange(0, d_model, 2).unsqueeze(0)
    enc = torch.zeros(seq_len, d_model)
    enc[:, 0::2] = torch.sin(pos / n ** (i / d_model))
    enc[:, 1::2] = torch.cos(pos / n ** (i / d_model))
    return enc

# alibi bias, see paper; may not help;
def alibi_bias(b, a = 1):
    b = torch.zeros_like(b)
    n = b.shape[0]//2 + 1
    for h in range(0, 8):
        bias = -1/2 ** (h + a + 1) * torch.arange(0, n)
        b[:n, h] = torch.flip(bias, [0])
        b[n-1:, h] = bias
    return b

# a patch that isn't a patch :(
class IdentityPatch(nn.Module):
    def __init__(self, 
        img_size,
            patch_size,
            in_chans,
            embed_dim, 
            bias = True,            
        ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

    def forward(self, x):
        return x

# seg model;
class SegmentationModel(nn.Module):
    def __init__(self, dims, 
                 encoder =  # shallow is better, mobilevit it is;
                #  'tu-tf_efficientnetv2_b3', #  'tu-tf_efficientnet_b3', 
                 'tu-mobilevitv2_075',
                 encoder_depth = 4,
                  dropout = 0.2, 
                  stack = False,
                  ):
        super().__init__()
        self.stack = stack
        self.mult = 2 ** (encoder_depth + 1)
        self.seg_model = smp.Unet(
                 encoder, 
                 in_channels = 1 if stack else 3,
                 encoder_depth = encoder_depth, 
                 decoder_channels = [256, 128, 64, 32, 16][:encoder_depth],
                 classes = dims,
                    **( {'encoder_weights': None} if OFFLINE else {}),
         )        
        # iterate over model, activate xformer dropout
        for name, module in self.seg_model.named_modules():
            if isinstance(module, nn.Dropout) and 'attn_drop' not in name:
                module.p = dropout
        # thin layers;
        self.seg_model.encoder.model.stages_2[1].transformer = nn.Sequential(
                    self.seg_model.encoder.model.stages_2[1].transformer[:2],
                )
        
    def forward(self, x):
        if self.stack:
            x = x.permute(0, 3, 1, 2)
            x = x.reshape(x.shape[0], -1, x.shape[-1]).unsqueeze(1).permute(0, 1, 3, 2)
        else:
            x = x.permute(0, 3, 2, 1)
            
        xo = x
        x = F.pad(x, (0, self.mult - x.shape[-1] % self.mult,)) 
        x = self.seg_model(x)
        x = x[:, :, :, :xo.shape[-1]]
        re_expand = xo.shape[-2] // x.shape[-2]
        if re_expand > 0:
            x = F.interpolate(x, scale_factor = (re_expand, 1), 
                                mode = 'bilinear')
        return x


# flipped channel ;
class GroupNorm1d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.norm = nn.GroupNorm(*args, **kwargs)
        
    def forward(self, x):
        assert len(x.shape) in [2, 3]
        x = x.permute(0, 2, 1) if len(x.shape) == 3 else x
        x = self.norm(x)
        x = x.permute(0, 2, 1) if len(x.shape) == 3 else x
        return x

# the core xformer/rnn backbone;
class WalkBackbone(nn.Module):
    def __init__(self, params, patch, seq, dims = 256, nheads = 8, 
                    act_layer = 'GELU',
                    dropout = 0.2, 
                    rnn = 'GRU', rnn_layers = 1, final_mult = 4, 
                    patch_dropout = 0.,
                    patch_act = 'Identity',
                    seg = False,
                    pre_norm = False,
                    patch_norm = False,
                    xformer_layers = 2, 
                    xformer_init_1 = 0.1,
                    xformer_init_2 = 0.1, 
                    xformer_init_scale = 1.,
                    xformer_attn_drop_rate = 0.1, xformer_drop_path_rate = 0.1,
                    deberta = False, h0 = False,
                    rel_pos = 'bias', alibi = True,                    
                    melspec = False, n_mels = None, mel_patch = True,
                    ):
        super().__init__()
        self.deberta = deberta 
        self.ch = (n_mels if melspec else 0) + (patch * (mel_patch if melspec else 1) )
        self.seg = seg
        self.seq = seq
        self.patch = patch
        self.pre_norm = pre_norm
        self.patch_norm = patch_norm
        if not deberta:
            self.xformer = (VisionTransformerRelPos if rel_pos 
                                    else VisionTransformer)(
                                                img_size = (seq, 1),
                                                patch_size = (1, 1),
                                                in_chans = dims,
                                                num_classes = 0,
                                                global_pool = '',
                                                embed_dim = dims,
                                                num_heads = nheads,
                                                embed_layer = IdentityPatch,
                                                act_layer = getattr(nn, act_layer),
                                                depth = xformer_layers,
                                                init_values = xformer_init_1,
                                                class_token = False,
                                                drop_rate = dropout,
                                                attn_drop_rate = xformer_attn_drop_rate,
                                                drop_path_rate = xformer_drop_path_rate,
                                                **({'rel_pos_type': rel_pos} if rel_pos else {})
                                                ) if xformer_layers > 0 else None
            if xformer_layers > 0 and rel_pos == 'bias' and alibi > 0:
               for i, b in enumerate(self.xformer.blocks):
                    b.attn.rel_pos.relative_position_bias_table.data = alibi_bias(
                        b.attn.rel_pos.relative_position_bias_table.data, alibi)
            if xformer_layers > 0 and not rel_pos: 
                self.xformer.pos_embed.data /= 2
                self.xformer.pos_embed.data[:] += 0.02 * getPositionalEncoding(seq, dims, 1000).unsqueeze(0)
            if xformer_layers > 0:
                for i, b in enumerate(self.xformer.blocks):
                    b.ls1.gamma.data[:] = torch.tensor(xformer_init_1 * xformer_init_scale ** i)
                    b.ls2.gamma.data[:] = torch.tensor(xformer_init_2 * xformer_init_scale ** i)
        else:
            # deberta v3
            
            # config = transformers.AutoConfig.from_pretrained('microsoft/deberta-v3-xsmall')
            cjson = '{"return_dict": true, "output_hidden_states": false, "output_attentions": false, "torchscript": false, "torch_dtype": null, "use_bfloat16": false, "tf_legacy_loss": false, "pruned_heads": {}, "tie_word_embeddings": true, "is_encoder_decoder": false, "is_decoder": false, "cross_attention_hidden_size": null, "add_cross_attention": false, "tie_encoder_decoder": false, "max_length": 20, "min_length": 0, "do_sample": false, "early_stopping": false, "num_beams": 1, "num_beam_groups": 1, "diversity_penalty": 0.0, "temperature": 1.0, "top_k": 50, "top_p": 1.0, "typical_p": 1.0, "repetition_penalty": 1.0, "length_penalty": 1.0, "no_repeat_ngram_size": 0, "encoder_no_repeat_ngram_size": 0, "bad_words_ids": null, "num_return_sequences": 1, "chunk_size_feed_forward": 0, "output_scores": false, "return_dict_in_generate": false, "forced_bos_token_id": null, "forced_eos_token_id": null, "remove_invalid_values": false, "exponential_decay_length_penalty": null, "suppress_tokens": null, "begin_suppress_tokens": null, "architectures": null, "finetuning_task": null, "id2label": {"0": "LABEL_0", "1": "LABEL_1"}, "label2id": {"LABEL_0": 0, "LABEL_1": 1}, "tokenizer_class": null, "prefix": null, "bos_token_id": null, "pad_token_id": 0, "eos_token_id": null, "sep_token_id": null, "decoder_start_token_id": null, "task_specific_params": null, "problem_type": null, "_name_or_path": "microsoft/deberta-v3-xsmall", "transformers_version": "4.27.1", "model_type": "deberta-v2", "position_buckets": 256, "norm_rel_ebd": "layer_norm", "share_att_key": true, "hidden_size": 384, "num_hidden_layers": 12, "num_attention_heads": 6, "intermediate_size": 1536, "hidden_act": "gelu", "hidden_dropout_prob": 0.1, "attention_probs_dropout_prob": 0.1, "max_position_embeddings": 512, "type_vocab_size": 0, "initializer_range": 0.02, "relative_attention": true, "max_relative_positions": -1, "position_biased_input": false, "pos_att_type": ["p2c", "c2p"], "vocab_size": 128100, "layer_norm_eps": 1e-07, "pooler_hidden_size": 384, "pooler_dropout": 0, "pooler_hidden_act": "gelu"}'
            config = transformers.DebertaV2Config.from_dict(json.loads(cjson))
            config.hidden_size = dims
            config.intermediate_size = dims * 4
            config.num_attention_heads = nheads
            config.num_hidden_layers = xformer_layers
            config.attention_probs_dropout_prob = xformer_attn_drop_rate
            config.hidden_dropout_prob = dropout
            self.xformer = transformers.models.deberta_v2.modeling_deberta_v2.DebertaV2Encoder(config)
            for i, b in enumerate(self.xformer.layer):
                b.attention.output.dense.weight.data *= xformer_init_1 * xformer_init_scale ** i
                b.intermediate.dense.weight.data *= xformer_init_2 * xformer_init_scale ** i

        # if xformer_layers == 0 or deberta:
        if seg:
            self.embed = nn.Sequential(SegmentationModel(**getParams(SegmentationModel, params)),
                                       nn.GroupNorm(4, dims),                                       
                                       )
        else:
            self.embed = nn.Sequential(PatchEmbed(img_size = (seq, 3),
                                                patch_size = (1, 3),
                                                in_chans = self.ch, 
                                                embed_dim = dims,
                                                bias = not self.patch_norm),
                                        GroupNorm1d(4, dims) if self.patch_norm else nn.Identity(),)
        self.norm = nn.LayerNorm(dims)
        self.patch_act = getattr(nn, patch_act)()
        self.dropout = nn.Dropout(dropout)
        self.patch_dropout = nn.Dropout1d(patch_dropout) 
        self.h0 = nn.Parameter(h0 * torch.randn(2 * rnn_layers, dims//2 * final_mult)) if h0 else None
        self.rnn = getattr(nn, rnn)(dims, dims//2 * final_mult,
                          num_layers = rnn_layers,
                          dropout = dropout if rnn_layers > 1 else 0, 
                          bidirectional = True, batch_first = True
                          ) if rnn is not None and rnn_layers > 0 else None
        
    def forward(self, x):    
        patch = self.patch
        attn = x[:, :, -patch:, :].reshape(x.shape[0], -1, patch * 3)
        attn = 1 * (attn.std(-1) > 1e-5)
        x = x.permute(0, 2, 1, 3)

        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = self.embed(x)
        x = self.patch_act(x)
        if self.seg: x = x.mean(-1).permute(0, 2, 1)
        if self.pre_norm: x = self.norm(x)
        x = x * attn.unsqueeze(-1)
        x = self.xformer(x, attn)['last_hidden_state'] if self.deberta else (
                self.xformer(x) if self.xformer else x)
        x = x * attn.unsqueeze(-1)
        if self.rnn is not None: 
            x = self.dropout(x)
            x = self.patch_dropout(x)
        xt = x
        x = self.rnn(x, self.h0.unsqueeze(1).repeat(1, x.shape[0], 1)
                             if torch.is_tensor(self.h0) else self.h0 
                     )[0] if self.rnn is not None else x
        if self.xformer is None:
            xt = x
        
        return x, xt

# it's a simple linear--norm/act/dropout;
class SimpleLinear(nn.Module):
    def __init__(self, dims, out_dims = None, dropout = 0.2, n_layers = 1):
        super().__init__()
        self.path = nn.Sequential()
        out_dims = out_dims or dims
        if n_layers == 0: 
            self.path.append(nn.Identity())
        for i in range(n_layers):                
            self.path.append(nn.Sequential(
                nn.Linear(dims if i == 0 else out_dims, out_dims, bias = False),
                nn.LayerNorm(out_dims),
                # GroupNorm1d(4, out_dims), ## tbd if better *** reverse if adverse
                nn.PReLU(), # *** experimental -- reverse if adverse
                # getattr(nn, act)(),
                nn.Dropout(dropout),
            ))
    
    def forward(self, x):
        return self.path(x)
    
# metadata use/infer, turn backbone into targets, adjust, etc etc.
class WalkNetwork(nn.Module):
    def __init__(self, params, seq, patch, dims = 256, nheads = 8, 
                    dropout = 0.2, final_dropout = 0.4,
                    mdims = 64, fdims = 64, 
                    frac_se = False, m_se = True,
                    len_se = False,
                    se_dims = 16, #se_act = 'Identity',
                    m_wt = 0.2, frac_wt = 0.3,

                    m_adj = True,
                    frac_adj = True,
                    
                    len_adj = False,

                    expanded = False,

                    m_gn = 0.5,
                    frac_gn = 0.1,
                    m_input_dropout = 0.3,                     
                    frac_input_dropout = 0.3,

                    m_adj_dropout = 0.2,
                    adj_dropout = 0.2,

                    m_adj_gn = 0.3,
                    adj_gn = 0.3,

                    fix_final = False,
                    # dual = False,
                    mae_xt = False, 
                    frac_pwr_mult = 2,
                    frac_rand = 0.5,
                    se_dropout = 0.3,
                    se_pact = 0.,

                    relabel = False,

                    ):
        super().__init__()
        self.num_meta = 16 if expanded else 12

        self.m_adj = m_adj
        self.frac_adj = frac_adj
        self.len_adj = len_adj

        self.m_gn = m_gn
        self.frac_gn = frac_gn

        self.m_adj_gn = m_adj_gn
        self.adj_gn = adj_gn

        self.mae_xt = mae_xt
        self.frac_pwr_mult = frac_pwr_mult
        self.frac_rand = frac_rand
        self.se_dims = se_dims
        self.se_dropout = se_dropout
        self.frac_se = frac_se
        # self.lin_se = lin_se
        self.m_se = m_se
        self.se_pact = se_pact

        self.fix_final = fix_final
        
        self.relabel = relabel

        
        self.backbone = WalkBackbone(params, **getParams(WalkBackbone, params))        

        rnn_dims = self.backbone.rnn.hidden_size * 2 if self.backbone.rnn is not None else dims
        self.mlinear = nn.Sequential(
            nn.Dropout(dropout),
            SimpleLinear(dims if self.backbone.xformer else rnn_dims, dims, dropout = dropout),
            nn.Linear(dims, self.num_meta)
        )
        self.final_dropout = nn.Dropout(final_dropout)
        self.ae_linear = nn.Sequential(SimpleLinear(dims if mae_xt else rnn_dims, dims * 2, 
                                                    dropout = final_dropout, ),
                                        nn.Linear(dims * 2, patch * 3))
        
        # linear adjutments
        self.meta_adjust = nn.Sequential(
            nn.Dropout(m_input_dropout),
            nn.Linear(self.num_meta, mdims, bias = False),
            GroupNorm1d(4, mdims, eps = 1e-3 ),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(mdims, 3, bias = False),
        )
        self.meta_adjust[-1].weight.data *= m_wt
        self.frac_adjust = nn.Sequential( 
            nn.Dropout(frac_input_dropout),
            nn.Linear(24, fdims, bias = False),
            GroupNorm1d(4, fdims),
            nn.PReLU(init = 0.8), 
            nn.Dropout(dropout),
            nn.Linear(fdims, 3, bias = False),
        )
        self.frac_adjust[-1].weight.data *= frac_wt
        self.adj_dropout = nn.Dropout(adj_dropout)
        self.m_adj_dropout = nn.Dropout(m_adj_dropout)
        
        if se_dims > 0: # PURELY LINEAR IN ALL VARIABLES-- __ -> f+mdims -> se_dims -> rnn_dims
            self.frac_linear = nn.Sequential(
                nn.Dropout(frac_input_dropout), 
                nn.Linear(24, fdims, bias = False), 
                # GroupNorm1d(4, fdims), 
                )
            self.meta = nn.Sequential(
                nn.Dropout(m_input_dropout), 
                nn.Linear(self.num_meta, mdims, bias = False), 
                # GroupNorm1d(4, mdims, eps = 1e-3 ),
                )
            
            # only medication and defog/tdcs, i.e. cardinality of 4; 
            self.meta_mask = nn.Parameter(torch.zeros((self.num_meta)),
                                                requires_grad = False)
            # self.meta_mask[2] = 1
            self.meta_mask[11] = 1 # keep in ONLY task type, not medication;
            
            # keep in frac and edge features, and possibly one length feature; 
            self.frac_mask = nn.Parameter(torch.tensor([0] * 24), # turned off (!)
                                                requires_grad = False)
              
            self.final_se = nn.Sequential( # linear(!) getattr(nn, se_act)(), # linear usually, 
                                nn.Dropout(se_dropout), nn.Linear(mdims + fdims, se_dims,),
                                nn.PReLU(init = se_pact) if se_pact > 0 else nn.Identity(), 
                                nn.Dropout(se_dropout), nn.Linear(se_dims, rnn_dims),                                        
                                nn.Dropout(se_dropout), nn.Sigmoid())
            
        self.final_bias1 = [-3, -1., -3,] + [0., 0., -1, -2, -1,] 
        self.final_bias2 = []# [2] + [0.] * 31  
                                                  
        self.final_linear = nn.Sequential(nn.Linear(rnn_dims, 
                                    patch * len(self.final_bias1 + self.final_bias2)
                                      * (2 if self.relabel else 1)
                                      ))
        if self.fix_final:
            w = self.final_linear[-1].weight.data 
            w = w.reshape(patch, -1, rnn_dims)
            w = w.mean(0, keepdim = True) * 2 + w/2
            self.final_linear[-1].weight.data = w.reshape_as(self.final_linear[-1].weight.data)
        self.final_linear[-1].bias.data = torch.tensor((self.final_bias1 + self.final_bias2)
                                                         * patch
                                                         * (2 if self.relabel else 1))

    def forward(self, x, m, frac, flen, adjust = True):    
        x, xt = self.backbone(x)
        ypm = self.mlinear(xt.mean(1))        
        yp_ae = self.ae_linear(self.final_dropout(x if not self.mae_xt else xt))
        # the frac/idx/len features
        # print(frac.shape)
        frac, lidx, ridx = frac[..., 0], frac[..., 1], frac[..., 2]
        flen = flen.reshape(len(flen), -1).expand_as(frac)
        assert frac.min() >=0 and frac.max() <= 1
        
        # all the frac/idx/len features
        has_frac = (frac[..., :10].std(-1) > 1e-5).float()
        svar = torch.stack([v ** (self.frac_pwr_mult ** i) - 0.5 + 0.1 * i
                                    for v in [frac, 1 - frac, 2 * (frac- 0.5).abs()]
                                    for i in range(3)]
                            + [( (flen/12500) ** (self.frac_pwr_mult ** -i) - 1 - (3.5 - i)/4 )
                                * (1 if self.len_adj else 0)
                                                for i in [1.5, 2.5, 3.5]]
                            , -1) * has_frac.unsqueeze(-1).unsqueeze(-1)
        # print(has_frac, has_frac.mean())
        svar = torch.cat([svar * m[..., 11].unsqueeze(1).unsqueeze(-1), 
                            svar * ( 1 - m[..., 11].unsqueeze(1).unsqueeze(-1) ) ], -1)
        
        # if random.random() < 1:#0.01:
        #     # print(lidx.mean())
        #     print(svar.mean([0, 1]).detach().cpu().numpy().round(2)[:12])
        #     print(svar.mean([0, 1]).detach().cpu().numpy().round(2)[12:])
        #     print(svar.std([0, 1]).detach().cpu().numpy().round(2)[:12])
        #     print(svar.std([0, 1]).detach().cpu().numpy().round(2)[12:])
        #     print()
        
        # expand m, add gn to both, 
        m = m.unsqueeze(1).repeat(1, svar.shape[1], 1)
        # print(m.shape, svar.shape)
        # print(m.std([0, 1]).detach().cpu().numpy().round(2))
        # print(svar.std([0, 1]).detach().cpu().numpy().round(2))
        if self.training:
            if self.m_gn > 0:    m = m + torch.randn_like(m) * self.m_gn
            if self.frac_gn > 0: svar = svar + torch.randn_like(svar) * self.frac_gn
        # print(m.std([0, 1]).detach().cpu().numpy().round(2))
        # print(svar.std([0, 1]).detach().cpu().numpy().round(2))

        # linear adjustments
        m_adj = self.meta_adjust(m)
        adj = self.frac_adjust(svar)
        # print(m_adj.std())
        # print(adj.std())
        # squeeze-excite final dims       
        # print(m.shape, svar.shape, x.shape) 
        if self.se_dims > 0 and adjust:
            fx = self.frac_linear(
                svar.reshape(svar.shape[0], x.shape[1], -1, svar.shape[-1]).mean(-2) * self.frac_mask)
            mx = self.meta(
                m.reshape(m.shape[0], x.shape[1], -1, self.num_meta)[..., 0, :] * self.meta_mask)
            # print(fx.std(), mx.std())
            se = self.final_se(torch.cat([fx * self.frac_se, mx * self.m_se], -1))
            # print(se.std())
            # print(x.shape, se.shape)
            if self.training: 
                se = torch.where(torch.rand_like(se.mean(-1)).unsqueeze(-1)
                                     < self.se_dropout, 0.5, se)
            x = x * se 
        else:
            x = x * 0.5

        yp = self.final_linear(self.final_dropout(x))
        yp, yp_ae = yp.reshape(yp.shape[0], frac.shape[1], -1), yp_ae.reshape(x.shape[0], frac.shape[1], 3)
        if adjust:
            yp[..., :3] += ( self.adj_dropout(adj) * self.frac_adj 
                            * has_frac.unsqueeze(-1).unsqueeze(-1) 
                            # to avoid using any positional info for dailies
                                    * ( (self.adj_gn * torch.randn_like(adj)).exp()
                                            if self.training else 1) 
                                + self.m_adj_dropout(m_adj) * self.m_adj 
                            * has_frac.unsqueeze(-1).unsqueeze(-1)
                            # to avoid learning that 0'ish values produce 0 targets/0 etc etc.;
                                    * ( (self.m_adj_gn * torch.randn_like(m_adj)).exp()
                                            if self.training else 1)
            )
        # print(self.final_se[2].weight.item() )        
        if random.random() < 1/500 and self.training: 
            print(adj.std([0, 1]).detach().float().cpu(), 
                    m_adj.std([0, 1]).detach().float().cpu(), 
                    (se.std([0, 1]).detach().float().cpu().mean() if self.se_dims > 0 else None),
                     (self.final_se[2].weight.item() if self.se_pact and self.se_dims > 0 else None))
        yp = torch.cat((yp[:, :, :len(self.final_bias1)].sigmoid(), 
                                    F.softmax(yp[:, :, len(self.final_bias1):], -1)), -1)

        return yp, yp_ae, ypm


class LossAvg:
    def __init__(self, module, rate = 4):
        self.module = module
        self.losses = defaultdict(float)                
        self.step = 0
        self.rate = rate

    def log(self, name, loss):
        a = min(1, self.rate / (self.step + self.rate))
        self.losses[name] *= (1 - a)
        self.losses[name] += a * loss.item()

    def print(self,):
        for k, v in self.losses.items():
            self.module.log(k, v, prog_bar = True)
        self.step += 1


class WalkModule(pl.LightningModule):
    def __init__(self, params, lr = 1e-4, weight_decay = 0.03, warmup = 0.05, 
                 smooth = 1e-3, steps = 1000, lion = False,
                 mae_divisor = 1., mmae_divisor = 30, 
                 v_wt = 0., 
                 aux_wt = 0.,
                 h_wt = 1., w_wt = 1.,
                 min_wt = 0., 
                #  mse_wt = 0., focal_wt = 1.,
                 focal_alpha = 0.25, focal_gamma = 1.5, 
                #  dual = False, 
                 fast_mult = 1, 

                 relabel = False,

                 ):
        super().__init__()
        self.params = SimpleNamespace(**locals())
        self.model = WalkNetwork(params, **getParams(WalkNetwork, params))
        self.base = nn.Parameter(torch.cat(
                        [torch.tensor(self.model.final_bias1).exp(),
                                    F.softmax(torch.tensor(self.model.final_bias2))]
                                    * (2 if relabel else 1)
                                    , -1)
                                           , requires_grad = False)
        self.weights = nn.Parameter(torch.tensor(([h_wt, 1., w_wt, ] 
                            + [v_wt] * 2 + [0.1 * v_wt] * (len(self.model.final_bias1) - 3 - 2)  
                            + [aux_wt] * len(self.model.final_bias2 ))
                                        * (2 if relabel else 1) 
                                                  ), requires_grad = False)
        self.losses = LossAvg(self)


    def forward(self, x, m, frac, flen, adjust = True):
        return self.model(x, m, frac, flen, adjust = adjust)
    
    def training_step(self, batch, batch_idx):
        x, y, s, frac, m, f, sidx, flen = batch
        b, seq, patch_img, ch = x.shape 
        yp, ypae, ypm = self(x, m, frac, flen)
        yps = yp * (1 - self.params.smooth) + self.params.smooth * self.base 

        # print(y.shape)
        n_task = len(self.model.final_bias2)
        y = torch.cat([y[..., :-1], 
                       F.one_hot(y[..., -1].long(), n_task )
                                if n_task > 0 else y[..., :0] # 0-size                        
                       ] * (2 if self.params.relabel else 1), -1) 
        
        # 
        mask = s.unsqueeze(-1).expand_as(yps).clone()    


        # mask out unlabeled items, in first run,...
        ul_items = torch.tensor(['unlabeled' in e for e in f], device = mask.device)
        if not self.params.relabel:
            mask *= ~ul_items.unsqueeze(-1).unsqueeze(-1)
            mask[..., 3:5] = 1
        else: # or mask the heads, in second run...
            mask[..., 3:5] = 1
            mask[..., 3 + yps.shape[-1]//2 
                    : 5 + yps.shape[-1]//2 ] = 1
            
            mask[..., :yps.shape[-1]//2] *= ~ul_items.unsqueeze(-1).unsqueeze(-1)
            mask[..., yps.shape[-1]//2:] *= ul_items.unsqueeze(-1).unsqueeze(-1)

        if random.random() < 1/100:#1/100:
            print(ul_items.float().mean(), mask.mean())

        mask = mask.clamp(min = self.params.min_wt)
        cls_loss = -(y * torch.log(yps) + (1 - y) * torch.log(1 - yps)
                        ) * self.weights * mask
        
        if self.params.v_wt > 0:
            self.losses.log('s_cls_loss', cls_loss[..., 3:5].mean() 
                            / self.weights[..., 3:5].mean())
        # if self.params.aux_wt > 0:
        #     self.losses.log('task_cls_loss', cls_loss[..., -n_task:].sum(-1).mean()   
        #             / self.weights[..., -n_task:].mean())
        
        self.losses.log('main_cls_loss', cls_loss[..., : yps.shape[-1] // 2].sum()
                                           / mask[..., : yps.shape[-1] // 2].mean(-1).sum() )
        self.losses.log('pl_cls_loss',   cls_loss[..., yps.shape[-1] // 2 :].sum()
                                           / mask[..., yps.shape[-1] // 2 :].mean(-1).sum() )

        cls_loss = cls_loss.sum() / mask.mean(-1).sum() / (2 if self.params.relabel else 1) 
        alpha = self.params.focal_alpha; gamma = self.params.focal_gamma; 
        focal_loss = -(y * torch.log(yps) * (1 - yp).pow(gamma) * alpha
                            + (1 - y) * torch.log(1 - yps) * yp.pow(gamma) * (1 - alpha)
                            ) * self.weights * mask#s.unsqueeze(-1)        
        focal_loss = focal_loss.sum() / mask.mean(-1).sum()  / (2 if self.params.relabel else 1)
        
        patch = ypae.shape[1] // seq
        x = x[:, :, -patch:, :].reshape(ypae.shape)
        mae_loss = ((ypae - x) / self.params.mae_divisor).pow(2).mean()        
        if batch_idx == 0:
            corrs = [round(torch.corrcoef(torch.stack((ypm[:, i].flatten(), m[:, i].flatten()))
                                            )[0, 1].item(), 4)                        
                            for i in range(ypm.shape[-1])]
            print(corrs)

        mmae_loss = ((ypm - m)/ self.params.mmae_divisor 
                            * (~ul_items.unsqueeze(-1))
                        ).pow(2).mean() 
        # mse_wt = self.params.mse_wt 
        focal_wt = 1# self.params.focal_wt
        loss = (#cls_loss * (1 - focal_wt) +
                 (5 * focal_wt * focal_loss )
                    #  * ( 1 - mse_wt) + mse_wt * mse_loss * 2
                + mae_loss + mmae_loss)
        # print(focal_loss); print(mae_loss); print(mmae_loss)
        # assert False
        self.losses.log('cls_loss', cls_loss)
        self.losses.log('focal_loss', focal_loss)
        self.losses.print()

        return loss
    
    def on_validation_epoch_start(self):
        self.yps, self.ys, self.ss, self.fs, self.idxs = [], [], [], [], []

    def validation_step(self, batch, batch_idx):
        x, y, s, frac, m, f, i, flen = batch
        yp, ypae, ypm = self(x, m, frac, flen)
        if random.random() < 1/100: print(yp)
        SAMPLE = 5
        self.yps.append(yp[:, ::SAMPLE, :3].cpu())
        self.ys.append(y[:, ::SAMPLE, :3].cpu())
        self.ss.append(s[:, ::SAMPLE].cpu())
        self.fs.append(f)
        self.idxs.append((i / SAMPLE).long())
        # if batch_idx == 0:
            # print(self.yps[0].shape)
            # print(self.ys[0].shape)
            # print(self.ss[0].shape)
            # print(self.fs[0])
            # print(self.idxs[0].shape)
        #     print(self.yps[0])
        #     print(self.ys[0])
        #     print(self.ss[0])
        #     print(self.fs[0])
        #     print(self.idxs[0])
        
    def on_validation_epoch_end(self):
        fs = np.concatenate(self.fs)
        idxs = torch.cat(self.idxs)
        uset = list(set(fs))

        slen = self.yps[0].shape[1]
        s = torch.cat(self.ss, 0).reshape(-1, slen, 1).repeat(1, 1, 3).cpu().numpy()
        ys = torch.cat(self.ys, 0).reshape(-1, slen, 3).cpu().numpy()
        yps = torch.cat(self.yps, 0).reshape(-1, slen, 3).cpu().numpy()        
        assert all([len(e) == len(yps) for e in [fs, idxs, s, ys, yps]])

        pred_dict, ct_dict, target_dict = {}, {}, {}
        for f in uset:
            l = 1000 * math.ceil( (idxs[fs == f].max() + slen) / 1000)
            pred_dict[f] = np.zeros((l, 3), dtype = np.float32)
            ct_dict[f] = np.zeros((l, 3), dtype = np.float32)
            target_dict[f] = np.zeros((l, 3), dtype = np.float32)

        for i in range(len(ys)):
            pred_dict[fs[i]][idxs[i]:idxs[i] + slen] += yps[i] * s[i]
            target_dict[fs[i]][idxs[i]:idxs[i] + slen] = ys[i] * s[i]
            ct_dict[fs[i]][idxs[i]:idxs[i] + slen] += s[i]

        self.pred_dict = pred_dict
        self.target_dict = target_dict        
        self.ct_dict = ct_dict

        final_ys, final_yps = [], []
        for k in ct_dict:
            f = ct_dict[k][:, 0] > 0
            final_ys.append(target_dict[k][f])
            final_yps.append(pred_dict[k][f] / ct_dict[k][f])
            assert (ct_dict[k].std(1) < 1e-5).all()
            assert f.sum() == 0 or ct_dict[k][f].min() >= 1

        final_ys, final_yps = np.concatenate(final_ys), np.concatenate(final_yps)
        if len(fs) < 1000: return 


        aps = []
        labels = 'htw'
        for i in range(3):
            aps.append(average_precision_score(final_ys[:, i], final_yps[:,  i]))
            self.log('val_ap_{}'.format(labels[i]), aps[-1], prog_bar = True)
        self.log('val_ap', np.mean(aps), prog_bar = True)


        s = torch.cat(self.ss, 0).flatten()
        ys = torch.cat(self.ys, 0).reshape(-1, 3)[s > 0].cpu().numpy()[::20]
        yps = torch.cat(self.yps, 0).reshape(-1, 3)[s > 0].cpu().numpy()[::20]

        aps = []
        labels = 'htw'
        for i in range(3):
            aps.append(average_precision_score(ys[:, i], yps[:,  i]))
        self.log('val_aps', np.mean(aps), prog_bar = True)

    def configure_optimizers(self):
        nd_keys = [k for k, v in self.model.named_parameters() 
                        if any([z in k for z in ['bias', '.bn', '.norm', 
                                                    '.ls', 'pos_embed',]])
                                                    or v.numel() < 50
                                                    ]
        fast_keys = [k for k, v in self.model.named_parameters() if '.rnn' in k and k not in nd_keys]
        print({k: v.shape for k, v in self.model.named_parameters() if k in nd_keys})
        print({k: v.shape for k, v in self.model.named_parameters() if k in fast_keys})

        lion = self.params.lion; lion_mult = 5
        mult = lion_mult if lion else 1
        lr = self.params.lr / mult
        wd = self.params.weight_decay * mult
        optimizer = (torch.optim.AdamW if not lion else Lion)(
                     [{'params': [v for k, v in self.model.named_parameters()
                                                    if k not in nd_keys + fast_keys],},
                     {'params': [v for k, v in self.model.named_parameters()
                                                    if k in fast_keys], 
                                                    'lr': lr / self.params.fast_mult,
                                                'weight_decay': wd * self.params.fast_mult,},                                 
                                       {'params': [v for k, v in self.model.named_parameters()
                                                    if k in nd_keys and k not in fast_keys], 
                                                'weight_decay': 0.,
                                                }],              
                                        lr = lr, 
                                weight_decay = wd, )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                            lambda x: min(1, x / min(1000, (self.params.warmup * self.params.steps)))
                                * (1/2 * (1 + math.cos(math.pi * x / self.params.steps))),
                                #  verbose = True 
                                 )
        return [optimizer], [{'scheduler': scheduler,
                              'interval': 'step'}]
    

