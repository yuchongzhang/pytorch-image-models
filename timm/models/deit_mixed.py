# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
from functools import partial
from typing import Optional, Type

import torch
from torch import nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import resample_abs_pos_embed
from timm.models.vision_transformer_mixed import MixedVisionTransformer, trunc_normal_, checkpoint_filter_fn
from ._builder import build_model_with_cfg
from ._registry import generate_default_cfgs, register_model, register_model_deprecations

__all__ = ['MixedVisionTransformerDistilled']  # model_registry will add each entrypoint fn to this


class MixedVisionTransformerDistilled(MixedVisionTransformer):
    """ Vision Transformer w/ Distillation Token and Head

    Distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, *args, **kwargs):
        weight_init = kwargs.pop('weight_init', '')
        super().__init__(*args, **kwargs, weight_init='skip')
        assert self.global_pool in ('token',)
        dd = {'device': kwargs.get('device', None), 'dtype': kwargs.get('dtype', None)}

        self.num_prefix_tokens = 2
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim, **dd))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + self.num_prefix_tokens, self.embed_dim, **dd))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes, **dd) if self.num_classes > 0 else nn.Identity()
        self.distilled_training = False  # must set this True to train w/ distillation token

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        trunc_normal_(self.dist_token, std=.02)
        super().init_weights(mode=mode)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed|dist_token',
            blocks=[
                (r'^blocks\.(\d+)', None),
                (r'^norm', (99999,))]  # final norm w/ last block
        )

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head, self.head_dist

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    @torch.jit.ignore
    def set_distilled_training(self, enable=True):
        self.distilled_training = enable

    def _pos_embed(self, x):
        if self.dynamic_img_size:
            B, H, W, C = x.shape
            prev_grid_size = self.patch_embed.grid_size
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                new_size=(H, W),
                old_size=prev_grid_size,
                num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.pos_embed
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            x = torch.cat((
                self.cls_token.expand(x.shape[0], -1, -1),
                self.dist_token.expand(x.shape[0], -1, -1),
                x),
                dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            x = torch.cat((
                self.cls_token.expand(x.shape[0], -1, -1),
                self.dist_token.expand(x.shape[0], -1, -1),
                x),
                dim=1)
            x = x + pos_embed
        return self.pos_drop(x)

    def forward_head(self, x, pre_logits: bool = False) -> torch.Tensor:
        x, x_dist = x[:, 0], x[:, 1]
        if pre_logits:
            return (x + x_dist) / 2
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.distilled_training and self.training and not torch.jit.is_scripting():
            # only return separate classification predictions when training in distilled mode
            return x, x_dist
        else:
            # during standard train / finetune, inference average the classifier predictions
            return (x + x_dist) / 2


def _create_deit(variant, pretrained=False, distilled=False, **kwargs):
    out_indices = kwargs.pop('out_indices', 3)
    model_cls = MixedVisionTransformerDistilled if distilled else MixedVisionTransformer
    model = build_model_with_cfg(
        model_cls,
        variant,
        pretrained,
        pretrained_filter_fn=partial(checkpoint_filter_fn, adapt_layer_scale=True),
        feature_cfg=dict(out_indices=out_indices, feature_cls='getter'),
        **kwargs,
    )
    return model


# =================== models for new sweep ============================

##################### DeiT Small models ###################
@register_model
def deit3_small_6L_patch16_224(pretrained=False, **kwargs) -> MixedVisionTransformer:
    """ DeiT-small model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, num_vanilla_heads=0, no_embed_class=True, init_values=1e-6, qk_norm=True)
    model = _create_deit('deit3_small_6L_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def deit3_small_5L_patch16_224(pretrained=False, **kwargs) -> MixedVisionTransformer:
    """ DeiT-small model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, num_vanilla_heads=1, no_embed_class=True, init_values=1e-6, qk_norm=True)
    model = _create_deit('deit3_small_5L_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def deit3_small_3L_patch16_224(pretrained=False, **kwargs) -> MixedVisionTransformer:
    """ DeiT-small model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, num_vanilla_heads=3, no_embed_class=True, init_values=1e-6, qk_norm=True)
    model = _create_deit('deit3_small_3L_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def deit3_small_1L_patch16_224(pretrained=False, **kwargs) -> MixedVisionTransformer:
    """ DeiT-small model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, num_vanilla_heads=5, no_embed_class=True, init_values=1e-6, qk_norm=True)
    model = _create_deit('deit3_small_1L_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

##################### DeiT Base models ####################
@register_model
def deit3_base_1L_patch16_224(pretrained=False, **kwargs) -> MixedVisionTransformer:
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, num_vanilla_heads=12-1, no_embed_class=True, init_values=1e-6, qk_norm=True)
    model = _create_deit('deit3_base_1L_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def deit3_base_3L_patch16_224(pretrained=False, **kwargs) -> MixedVisionTransformer:
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, num_vanilla_heads=12-3, no_embed_class=True, init_values=1e-6, qk_norm=True)
    model = _create_deit('deit3_base_3L_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def deit3_base_6L_patch16_224(pretrained=False, **kwargs) -> MixedVisionTransformer:
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, num_vanilla_heads=12-6, no_embed_class=True, init_values=1e-6, qk_norm=True)
    model = _create_deit('deit3_base_6L_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def deit3_base_9L_patch16_224(pretrained=False, **kwargs) -> MixedVisionTransformer:
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, num_vanilla_heads=12-9, no_embed_class=True, init_values=1e-6, qk_norm=True)
    model = _create_deit('deit3_base_9L_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def deit3_base_11L_patch16_224(pretrained=False, **kwargs) -> MixedVisionTransformer:
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, num_vanilla_heads=12-11, no_embed_class=True, init_values=1e-6, qk_norm=True)
    model = _create_deit('deit3_base_11L_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def deit3_base_12L_patch16_224(pretrained=False, **kwargs) -> MixedVisionTransformer:
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, num_vanilla_heads=12-12, no_embed_class=True, init_values=1e-6, qk_norm=True)
    model = _create_deit('deit3_base_12L_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


##################### DeiT Large models #####################
@register_model
def deit3_large_2L_patch16_192(pretrained=False, **kwargs) -> MixedVisionTransformer:
    model_args = dict(img_size=192, patch_size=16, embed_dim=1024, depth=24, num_heads=16, num_vanilla_heads=16-2, no_embed_class=True, init_values=1e-6, qk_norm=True)
    model = _create_deit('deit3_large_2L_patch16_192', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def deit3_large_4L_patch16_192(pretrained=False, **kwargs) -> MixedVisionTransformer:
    model_args = dict(img_size=192, patch_size=16, embed_dim=1024, depth=24, num_heads=16, num_vanilla_heads=16-4, no_embed_class=True, init_values=1e-6, qk_norm=True)
    model = _create_deit('deit3_large_4L_patch16_192', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def deit3_large_6L_patch16_192(pretrained=False, **kwargs) -> MixedVisionTransformer:
    model_args = dict(img_size=192, patch_size=16, embed_dim=1024, depth=24, num_heads=16, num_vanilla_heads=16-6, no_embed_class=True, init_values=1e-6, qk_norm=True)
    model = _create_deit('deit3_large_6L_patch16_192', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def deit3_large_8L_patch16_192(pretrained=False, **kwargs) -> MixedVisionTransformer:
    model_args = dict(img_size=192, patch_size=16, embed_dim=1024, depth=24, num_heads=16, num_vanilla_heads=16-8, no_embed_class=True, init_values=1e-6, qk_norm=True)
    model = _create_deit('deit3_large_8L_patch16_192', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def deit3_large_10L_patch16_192(pretrained=False, **kwargs) -> MixedVisionTransformer:
    model_args = dict(img_size=192, patch_size=16, embed_dim=1024, depth=24, num_heads=16, num_vanilla_heads=16-10, no_embed_class=True, init_values=1e-6, qk_norm=True)
    model = _create_deit('deit3_large_10L_patch16_192', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def deit3_large_12L_patch16_192(pretrained=False, **kwargs) -> MixedVisionTransformer:
    model_args = dict(img_size=192, patch_size=16, embed_dim=1024, depth=24, num_heads=16, num_vanilla_heads=16-12, no_embed_class=True, init_values=1e-6, qk_norm=True)
    model = _create_deit('deit3_large_12L_patch16_192', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def deit3_large_14L_patch16_192(pretrained=False, **kwargs) -> MixedVisionTransformer:
    model_args = dict(img_size=192, patch_size=16, embed_dim=1024, depth=24, num_heads=16, num_vanilla_heads=16-14, no_embed_class=True, init_values=1e-6, qk_norm=True)
    model = _create_deit('deit3_large_14L_patch16_192', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def deit3_large_16L_patch16_192(pretrained=False, **kwargs) -> MixedVisionTransformer:
    model_args = dict(img_size=192, patch_size=16, embed_dim=1024, depth=24, num_heads=16, num_vanilla_heads=16-16, no_embed_class=True, init_values=1e-6, qk_norm=True)
    model = _create_deit('deit3_large_16L_patch16_192', pretrained=pretrained, **dict(model_args, **kwargs))
    return model
