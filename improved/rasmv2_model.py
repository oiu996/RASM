# -*- coding: utf-8 -*-
"""RASMv2: 在既有 RASM 之上以"新增"方式构建的改进模型。

拓扑、属性命名与既有 RASM 完全一致 (encoderlayer_0/1/2 -> conv -> decoderlayer_0/1/2)，
因此 preset='rasm' 时可直接加载官方发布的 RASM 权重，作为消融实验的基线。

新增内容(相对论文):
  A. MGRA    掩码引导的区域注意力: RIB + NS-Gate + dilation 感知 RPB
  B. NSRC    非阴影区域上下文聚合
  C. IAFM    逐层 Mask-FiLM
  D. 感受野尺度自适应 (dilations='adaptive')
  E. 残差门控: 只在阴影(+半影)区域放行残差，非阴影区仅允许常值级光照校正
"""

import os
import sys

_RASM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RASM_ROOT not in sys.path:
    sys.path.insert(0, _RASM_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from model import Downsample, Upsample, InputProj, OutputProj
from improved.rasmv2_modules import BasicRASMv2

# ------------------------------------------------------------------------------
# 消融预设: 每套预设只开一类改进，便于逐个验证贡献
# ------------------------------------------------------------------------------
PRESETS = {
    # 基线: 等价于论文 RASM(仅把 natten 换成等价的纯 PyTorch 实现)，可加载官方权重
    'rasm' : dict(mask_bias=False, ns_gate=False, use_nsrc=False, use_film=False,
                  residual_gate=False, attn_impl='unfold'),
    'mgra' : dict(mask_bias=True,  ns_gate=True,  use_nsrc=False, use_film=False,
                  residual_gate=False, attn_impl='unfold'),
    'nsrc' : dict(mask_bias=False, ns_gate=False, use_nsrc=True,  use_film=False,
                  residual_gate=False, attn_impl='unfold'),
    'film' : dict(mask_bias=False, ns_gate=False, use_nsrc=False, use_film=True,
                  residual_gate=False, attn_impl='unfold'),
    'rgate': dict(mask_bias=False, ns_gate=False, use_nsrc=False, use_film=False,
                  residual_gate=True,  attn_impl='unfold'),
    'full' : dict(mask_bias=True,  ns_gate=True,  use_nsrc=True,  use_film=True,
                  residual_gate=True,  attn_impl='unfold'),
}


def _resolve(preset, **kwargs):
    """preset 提供默认值，显式传入的参数优先。"""
    cfg = dict(PRESETS.get(preset, PRESETS['full']))
    for k, v in kwargs.items():
        if v is not None:
            cfg[k] = v
    return cfg


def _inv_softplus(x):
    import math
    return math.log(math.expm1(x)) if x < 20.0 else x


class RASMv2(nn.Module):
    def __init__(self, img_size=256, in_chans=3,
                 embed_dim=32, depths=[2, 2, 2, 2, 2, 2, 2, 2, 2],
                 num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
                 win_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, token_projection='linear', token_mlp='leff',
                 se_layer=False, dowsample=Downsample, upsample=Upsample,
                 # ---------------- v2 新增开关 ----------------
                 preset='full', kernel_size=11, dilations=(2, 2, 2),
                 nsrc_kernel=11, nsrc_dilation=2,
                 mask_bias=None, ns_gate=None, ns_gate_hard=False, ns_gate_init=0.5,
                 use_nsrc=None, use_film=None, residual_gate=None,
                 res_gate_kernel=9, res_gate_beta=0.1,
                 attn_impl=None, border_mode='zero', chunk_elements=2_000_000,
                 **kwargs):
        super().__init__()
        cfg = _resolve(preset, mask_bias=mask_bias, ns_gate=ns_gate, use_nsrc=use_nsrc,
                       use_film=use_film, residual_gate=residual_gate, attn_impl=attn_impl)
        self.preset = preset
        self.cfg = cfg

        if dilations == 'adaptive':
            # 让三个区域注意力层的"输入坐标系感受野"大致相同:
            # 1/4 分辨率(跨度 20*d*4) 用 d=2，1/8 分辨率(跨度 20*d*8) 用 d=1
            dilations = (2, 1, 2)
        if isinstance(dilations, (int, float)):
            dilations = (int(dilations),) * 3
        dilations = tuple(dilations)

        self.num_enc_layers = len(depths) // 2
        self.num_dec_layers = len(depths) // 2
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.token_projection = token_projection
        self.mlp = token_mlp
        self.win_size = win_size
        self.reso = img_size
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.kernel_size = kernel_size
        self.dilations = dilations

        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))]
        conv_dpr = [drop_path_rate] * depths[4]
        dec_dpr = enc_dpr[::-1]

        # v2 的 block 级参数（三个区域注意力层共用）; kernel_size/dilations 由外层显式传入
        ra_kwargs = dict(use_nsrc=cfg['use_nsrc'],
                         nsrc_kernel=nsrc_kernel, nsrc_dilation=nsrc_dilation,
                         use_film=cfg['use_film'], mask_bias=cfg['mask_bias'],
                         ns_gate=cfg['ns_gate'], ns_gate_hard=ns_gate_hard,
                         ns_gate_init=ns_gate_init, attn_impl=cfg['attn_impl'],
                         border_mode=border_mode, chunk_elements=chunk_elements)
        ca_kwargs = dict(use_film=cfg['use_film'])

        # Input / Output
        self.input_proj = InputProj(in_channel=4, out_channel=embed_dim, kernel_size=3, stride=1,
                                    act_layer=nn.LeakyReLU)
        self.output_proj = OutputProj(in_channel=2 * embed_dim, out_channel=in_chans, kernel_size=3, stride=1)

        # Encoder
        self.encoderlayer_0 = BasicRASMv2(dim=embed_dim, output_dim=embed_dim,
                                          input_resolution=(img_size, img_size), depth=depths[0],
                                          num_heads=num_heads[0], win_size=win_size,
                                          mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          drop=drop_rate, attn_drop=attn_drop_rate,
                                          drop_path=enc_dpr[sum(depths[:0]):sum(depths[:1])],
                                          norm_layer=norm_layer, use_checkpoint=use_checkpoint,
                                          token_projection=token_projection, token_mlp=token_mlp,
                                          se_layer=se_layer, cab=True, block_kwargs=ca_kwargs)
        self.dowsample_0 = dowsample(embed_dim, embed_dim * 2)
        self.encoderlayer_1 = BasicRASMv2(dim=embed_dim * 2, output_dim=embed_dim * 2,
                                          input_resolution=(img_size // 2, img_size // 2), depth=depths[1],
                                          num_heads=num_heads[1], win_size=win_size,
                                          mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          drop=drop_rate, attn_drop=attn_drop_rate,
                                          drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])],
                                          norm_layer=norm_layer, use_checkpoint=use_checkpoint,
                                          token_projection=token_projection, token_mlp=token_mlp,
                                          se_layer=se_layer, cab=True, block_kwargs=ca_kwargs)
        self.dowsample_1 = dowsample(embed_dim * 2, embed_dim * 4)
        self.encoderlayer_2 = BasicRASMv2(dim=embed_dim * 4, output_dim=embed_dim * 4,
                                          input_resolution=(img_size // 4, img_size // 4), depth=depths[2],
                                          num_heads=num_heads[2], win_size=win_size,
                                          mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          drop=drop_rate, attn_drop=attn_drop_rate,
                                          drop_path=enc_dpr[sum(depths[:2]):sum(depths[:3])],
                                          norm_layer=norm_layer, use_checkpoint=use_checkpoint,
                                          token_projection=token_projection, token_mlp=token_mlp,
                                          se_layer=se_layer, cab=False,
                                          kernel_size=kernel_size, dilations=(dilations[0],),
                                          block_kwargs=ra_kwargs)
        self.dowsample_2 = dowsample(embed_dim * 4, embed_dim * 8)

        # Bottleneck
        self.conv = BasicRASMv2(dim=embed_dim * 8, output_dim=embed_dim * 8,
                                input_resolution=(img_size // 8, img_size // 8), depth=depths[4],
                                num_heads=num_heads[4], win_size=win_size,
                                mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=conv_dpr,
                                norm_layer=norm_layer, use_checkpoint=use_checkpoint,
                                token_projection=token_projection, token_mlp=token_mlp,
                                se_layer=se_layer, cab=False,
                                kernel_size=kernel_size, dilations=(dilations[1],),
                                block_kwargs=ra_kwargs)

        # Decoder
        self.upsample_0 = upsample(embed_dim * 8, embed_dim * 4)
        self.decoderlayer_0 = BasicRASMv2(dim=embed_dim * 8, output_dim=embed_dim * 8,
                                          input_resolution=(img_size // 4, img_size // 4), depth=depths[6],
                                          num_heads=num_heads[6], win_size=win_size,
                                          mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          drop=drop_rate, attn_drop=attn_drop_rate,
                                          drop_path=dec_dpr[sum(depths[5:6]):sum(depths[5:7])],
                                          norm_layer=norm_layer, use_checkpoint=use_checkpoint,
                                          token_projection=token_projection, token_mlp=token_mlp,
                                          se_layer=se_layer, cab=False,
                                          kernel_size=kernel_size, dilations=(dilations[2],),
                                          block_kwargs=ra_kwargs)
        self.upsample_1 = upsample(embed_dim * 8, embed_dim * 2)
        self.decoderlayer_1 = BasicRASMv2(dim=embed_dim * 4, output_dim=embed_dim * 4,
                                          input_resolution=(img_size // 2, img_size // 2), depth=depths[7],
                                          num_heads=num_heads[7], win_size=win_size,
                                          mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          drop=drop_rate, attn_drop=attn_drop_rate,
                                          drop_path=dec_dpr[sum(depths[5:7]):sum(depths[5:8])],
                                          norm_layer=norm_layer, use_checkpoint=use_checkpoint,
                                          token_projection=token_projection, token_mlp=token_mlp,
                                          se_layer=se_layer, cab=True, block_kwargs=ca_kwargs)
        self.upsample_2 = upsample(embed_dim * 4, embed_dim)
        self.decoderlayer_2 = BasicRASMv2(dim=embed_dim * 2, output_dim=embed_dim * 2,
                                          input_resolution=(img_size, img_size), depth=depths[8],
                                          num_heads=num_heads[8], win_size=win_size,
                                          mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          drop=drop_rate, attn_drop=attn_drop_rate,
                                          drop_path=dec_dpr[sum(depths[5:8]):sum(depths[5:9])],
                                          norm_layer=norm_layer, use_checkpoint=use_checkpoint,
                                          token_projection=token_projection, token_mlp=token_mlp,
                                          se_layer=se_layer, cab=True, block_kwargs=ca_kwargs)

        # E. 残差门控
        self.residual_gate = cfg['residual_gate']
        if self.residual_gate:
            self.res_gate_kernel = res_gate_kernel
            self.res_gate_raw = nn.Parameter(torch.full((1,), _inv_softplus(res_gate_beta)))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def extra_repr(self):
        return (f'embed_dim={self.embed_dim}, token_projection={self.token_projection}, '
                f'token_mlp={self.mlp}, win_size={self.win_size}, '
                f'kernel={self.kernel_size}, dilations={self.dilations}, preset={self.preset}')

    def forward(self, x, xm, mask=None):
        # 输入投影: [RGB + mask] -> 特征
        xi = torch.cat((x, xm), dim=1)
        self.img_size = (x.shape[2], x.shape[3])
        y = self.input_proj(xi)
        y = self.pos_drop(y)

        # Encoder
        conv0 = self.encoderlayer_0(y, xm, mask=mask, img_size=self.img_size)
        pool0 = self.dowsample_0(conv0, img_size=self.img_size)
        m = nn.MaxPool2d(2)
        xm1 = m(xm)
        self.img_size = (int(self.img_size[0] / 2), int(self.img_size[1] / 2))
        conv1 = self.encoderlayer_1(pool0, xm1, mask=mask, img_size=self.img_size)
        pool1 = self.dowsample_1(conv1, img_size=self.img_size)
        xm2 = m(xm1)
        self.img_size = (int(self.img_size[0] / 2), int(self.img_size[1] / 2))
        conv2 = self.encoderlayer_2(pool1, xm2, mask=mask, img_size=self.img_size)
        pool2 = self.dowsample_2(conv2, img_size=self.img_size)
        self.img_size = (int(self.img_size[0] / 2), int(self.img_size[1] / 2))
        xm3 = m(xm2)

        # Bottleneck
        conv3 = self.conv(pool2, xm3, mask=mask, img_size=self.img_size)

        # Decoder
        up0 = self.upsample_0(conv3, img_size=self.img_size)
        self.img_size = (int(self.img_size[0] * 2), int(self.img_size[1] * 2))
        deconv0 = torch.cat([up0, conv2], -1)
        deconv0 = self.decoderlayer_0(deconv0, xm2, mask=mask, img_size=self.img_size)

        up1 = self.upsample_1(deconv0, img_size=self.img_size)
        self.img_size = (int(self.img_size[0] * 2), int(self.img_size[1] * 2))
        deconv1 = torch.cat([up1, conv1], -1)
        deconv1 = self.decoderlayer_1(deconv1, xm1, mask=mask, img_size=self.img_size)

        up2 = self.upsample_2(deconv1, img_size=self.img_size)
        self.img_size = (int(self.img_size[0] * 2), int(self.img_size[1] * 2))
        deconv2 = torch.cat([up2, conv0], -1)
        deconv2 = self.decoderlayer_2(deconv2, xm, mask=mask, img_size=self.img_size)

        # Output
        res = self.output_proj(deconv2, img_size=self.img_size)
        if self.residual_gate:
            beta = F.softplus(self.res_gate_raw).view(1, 1, 1, 1)
            md = F.max_pool2d(xm, self.res_gate_kernel, 1, self.res_gate_kernel // 2)
            res = res * (md + beta * (1.0 - md))
        return res + x


def build_rasmv2(opt):
    """与既有 utils.get_arch(opt) 同签名的构造入口（供 train_v2 / test_v2 使用）。"""
    return RASMv2(img_size=opt.train_ps, embed_dim=opt.embed_dim, win_size=opt.win_size,
                  token_projection=opt.token_projection, token_mlp=opt.token_mlp,
                  preset=getattr(opt, 'preset', 'full'),
                  kernel_size=getattr(opt, 'kernel_size', 11),
                  dilations=getattr(opt, 'dilations', (2, 2, 2)),
                  attn_impl=getattr(opt, 'attn_impl', None),
                  border_mode=getattr(opt, 'border_mode', 'zero'))


def load_weights(model, weights, strict=False):
    """加载权重。RASMv2 相对官方 RASM 有新增/删除的参数，故默认 strict=False 并打印差异。"""
    ckpt = torch.load(weights, map_location='cpu')
    state = ckpt.get('state_dict', ckpt.get('model', ckpt))
    state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}
    out = model.load_state_dict(state, strict=strict)
    if out.missing_keys:
        print(f'[load_weights] 缺失(将随机初始化) {len(out.missing_keys)} 项, '
              f'如: {out.missing_keys[:3]}')
    if out.unexpected_keys:
        print(f'[load_weights] 未使用(官方旧结构) {len(out.unexpected_keys)} 项, '
              f'如: {out.unexpected_keys[:3]}')
    return out
