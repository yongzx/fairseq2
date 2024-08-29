# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.wav2vec2.asr.factory import Wav2Vec2AsrConfig, wav2vec2_asr_arch
from fairseq2.models.wav2vec2.factory import wav2vec2_encoder_archs


@wav2vec2_asr_arch("base_10h")
def _base_10h() -> Wav2Vec2AsrConfig:
    return Wav2Vec2AsrConfig()


@wav2vec2_asr_arch("base_100h")
def _base_100h() -> Wav2Vec2AsrConfig:
    config = _base_10h()

    config.encoder_config.layer_drop_p = 0.1

    return config


@wav2vec2_asr_arch("large_10h")
def _large_10h() -> Wav2Vec2AsrConfig:
    config = _base_10h()

    config.encoder_config = wav2vec2_encoder_archs.get("large")
    config.encoder_config.feature_gradient_scale = 1.0
    config.encoder_config.dropout_p = 0.0
    config.encoder_config.attn_dropout_p = 0.0
    config.encoder_config.ffn_inner_dropout_p = 0.1
    config.encoder_config.layer_drop_p = 0.1

    config.max_temporal_mask_prob = 0.80
    config.max_spatial_mask_prob = 0.30

    return config


@wav2vec2_asr_arch("large_100h")
def _large_100h() -> Wav2Vec2AsrConfig:
    config = _large_10h()

    config.max_temporal_mask_prob = 0.53
    config.max_spatial_mask_prob = 0.55

    return config


@wav2vec2_asr_arch("large_lv60k_10h")
def _large_lv60k_10h() -> Wav2Vec2AsrConfig:
    config = _base_10h()

    config.encoder_config = wav2vec2_encoder_archs.get("large_lv60k")
    config.encoder_config.feature_gradient_scale = 1.0
    config.encoder_config.dropout_p = 0.0
    config.encoder_config.attn_dropout_p = 0.0
    config.encoder_config.ffn_inner_dropout_p = 0.1
    config.encoder_config.layer_drop_p = 0.1

    config.max_temporal_mask_prob = 0.80
    config.max_spatial_mask_prob = 0.30

    return config

#############################################################################
@wav2vec2_asr_arch("mms_base_300m_asr")
def _mms_base_300m_eng_accent() -> Wav2Vec2AsrConfig:
    config = _large_lv60k_10h()

    #### from /private/home/yongzx/mms/gh/yong_mms/debug/mms300m_config.yaml
    #     model:
    #   _name: wav2vec2
    #   activation_dropout: 0.0
    #   activation_fn: gelu
    #   attention_dropout: 0.0
    #   checkpoint_activations: false
    #   codebook_negatives: 0
    #   conv_bias: true
    #   conv_feature_layers: '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]'
    #   conv_pos: 128
    #   conv_pos_groups: 16
    #   cross_sample_negatives: 0
    #   dropout: 0.0
    #   dropout_features: 0.0 # (dropout to apply to the features (after feat extr))
    #   dropout_input: 0.0 # (dropout to apply to the input (after feat extr))
    #   encoder_attention_heads: 16
    #   encoder_embed_dim: 1024
    #   encoder_ffn_embed_dim: 4096
    #   encoder_layerdrop: 0.0
    #   encoder_layers: 24
    #   extractor_mode: layer_norm
    #   feature_grad_mult: 1.0
    #   final_dim: 768
    #   latent_dim: 0
    #   latent_groups: 2
    #   latent_temp:
    #   - 2.0
    #   - 0.1
    #   - 0.999995
    #   latent_vars: 320
    #   layer_norm_first: true
    #   logit_temp: 0.1
    #   mask_channel_length: 10
    #   mask_channel_min_space: 1
    #   mask_channel_other: 0.0
    #   mask_channel_prob: 0.0
    #   mask_channel_selection: static
    #   mask_length: 10  # ("mask length")
    #   mask_min_space: 1
    #   mask_other: 0.0
    #   mask_prob: 0.65
    #   mask_selection: static
    #   negatives_from_everywhere: false
    #   no_mask_channel_overlap: false
    #   no_mask_overlap: false
    #   num_negatives: 100
    #   offload_activations: false
    #   quantize_input: false
    #   quantize_targets: true
    #   same_quantizer: false
    #   target_glu: false

    ##### from: /checkpoint/vineelkpratap/trash/fairseq-py/examples/wav2vec/config/finetuning/mmasr/train_bible_mono_ft2.yaml
    ##### model:
    #   _name: wav2vec_ctc
    #   w2v_path: /checkpoint/arbabu/XLSR2/model_versions/bible/bible_og/checkpoint_3_1000000_fixed.pt
    #   apply_mask: true
    #   mask_prob: 0.3  # (probability of replacing a token with mask)
    #   mask_channel_prob: 0.0 # (probability of replacing a feature with 0)
    #   mask_channel_length: 5 # (length of the mask for features (channels))
    #   layerdrop: 0.1
    #   activation_dropout: 0.1
    #   feature_grad_mult: 0.0
    #   freeze_finetune_updates: 0  # --> recipe 

    config.encoder_config.first_pass_dropout_p = 0.0 # dropout_features
    config.encoder_config.layer_norm_features = True # extractor_mode: layer_norm
    config.encoder_config.feature_gradient_scale = 0.0 # feature_grad_mult
    config.encoder_config.dropout_p = 0.0 # dropout
    config.encoder_config.attn_dropout_p = 0.0  # attention_dropout
    config.encoder_config.ffn_inner_dropout_p = 0.1  # activation_dropout
    config.encoder_config.layer_drop_p = 0.1  # layerdrop
    config.max_temporal_mask_prob = 0.3  # mask_prob
    config.temporal_mask_span_len = 0.3  # mask_length
    config.max_spatial_mask_prob = 0 # mask_channel_prob
    config.spatial_mask_span_len = 5 # mask_channel_length (?)

    return config
#############################################################################

@wav2vec2_asr_arch("large_lv60k_100h")
def _large_lv60k_100h() -> Wav2Vec2AsrConfig:
    config = _large_lv60k_10h()

    config.max_temporal_mask_prob = 0.53
    config.max_spatial_mask_prob = 0.55

    return config
