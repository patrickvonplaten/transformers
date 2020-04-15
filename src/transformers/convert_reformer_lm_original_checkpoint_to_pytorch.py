#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert Reformer checkpoint."""

import argparse
import logging
import pickle

import numpy as np

import torch
from tensorflow.compat.v1.io.gfile import GFile

from transformers import ReformerConfig, ReformerModelWithLMHead

logging.basicConfig(level=logging.INFO)


def _set_param(torch_layer, weight, bias=None):
    assert torch_layer.weight.shape == weight.shape, "{} layer.weight does not match".format(torch_layer)
    torch_layer.weight = torch.nn.Parameter(weight)
    if bias is not None:
        assert torch_layer.bias.shape == bias.shape, "{} layer.bias does not match".format(torch_layer)
        torch_layer.bias = torch.nn.Parameter(bias)


def _set_layer_weights_in_torch_lsh(weights, torch_layer, hidden_size):
    # set torch weights for 1-to-1 comparison
    np_query_key = np.asarray(weights[0])
    np_value = np.asarray(weights[1])
    np_dense = np.asarray(weights[2])

    _set_param(torch_layer.self_attention.query_key, torch.tensor(np_query_key).transpose(1, 2).contiguous().view(-1, hidden_size))
    _set_param(torch_layer.self_attention.value, torch.tensor(np_value).transpose(1, 2).contiguous().view(-1, hidden_size))
    _set_param(torch_layer.output.dense, torch.tensor(np_dense).view(-1, hidden_size).contiguous().transpose(0, 1))


def _set_layer_weights_in_torch_local(weights, torch_layer, hidden_size):
    # set torch weights for 1-to-1 comparison
    np_query = np.asarray(weights[0])
    np_key = np.asarray(weights[1])
    np_value = np.asarray(weights[2])
    np_dense = np.asarray(weights[3])

    _set_param(torch_layer.self_attention.query, torch.tensor(np_query).transpose(1, 2).contiguous().view(-1, hidden_size))
    _set_param(torch_layer.self_attention.key, torch.tensor(np_key).transpose(1, 2).contiguous().view(-1, hidden_size))
    _set_param(torch_layer.self_attention.value, torch.tensor(np_value).transpose(1, 2).contiguous().view(-1, hidden_size))
    _set_param(torch_layer.output.dense, torch.tensor(np_dense).view(-1, hidden_size).contiguous().transpose(0, 1))


def _set_block_weights_in_torch(weights, torch_block, hidden_size):
    # layernorm 1
    layer_norm_1 = weights[0][0][0]
    layer_norm_1_weight = np.asarray(layer_norm_1[0])
    layer_norm_1_bias = np.asarray(layer_norm_1[1])
    _set_param(torch_block.attention.layer_norm, torch.tensor(layer_norm_1_weight), torch.tensor(layer_norm_1_bias))

    # lsh or local weights + output
    attn_weights = weights[0][1]
    if len(attn_weights) < 4:
        _set_layer_weights_in_torch_lsh(attn_weights, torch_block.attention, hidden_size)
    else:
        _set_layer_weights_in_torch_local(attn_weights, torch_block.attention, hidden_size)

    # intermediate weighs
    intermediate_weights = weights[2][0][2][2]

    # Chunked Feed Forward
    if len(intermediate_weights) == 4:
        intermediate_weights = intermediate_weights[2]

    # layernorm 2
    layer_norm_2_weight = np.asarray(intermediate_weights[0][0])
    layer_norm_2_bias = np.asarray(intermediate_weights[0][1])
    _set_param(torch_block.feed_forward.layer_norm, torch.tensor(layer_norm_2_weight), torch.tensor(layer_norm_2_bias))

    # intermediate dense
    inter_dense_weight = np.asarray(intermediate_weights[1][0])
    inter_dense_bias = np.asarray(intermediate_weights[1][1])
    _set_param(torch_block.feed_forward.dense.dense, torch.tensor(inter_dense_weight).transpose(0, 1).contiguous(), torch.tensor(inter_dense_bias))

    # intermediate out
    out_dense_weight = np.asarray(intermediate_weights[4][0])
    out_dense_bias = np.asarray(intermediate_weights[4][1])
    _set_param(torch_block.feed_forward.output.dense, torch.tensor(out_dense_weight).transpose(0, 1).contiguous(), torch.tensor(out_dense_bias))


def _set_model_weights_in_torch(weights, torch_model, hidden_size):
    # reformer model
    torch_model_reformer = torch_model.reformer

    # word embeds
    word_embeddings = np.asarray(weights[1])
    _set_param(torch_model_reformer.embeddings.word_embeddings, torch.tensor(word_embeddings))

    # axial position encodings
    if isinstance(weights[3], tuple):
        position_embeddings = torch_model_reformer.embeddings.position_embeddings
        for emb_idx in range(len(position_embeddings.weights)):
            emb_weights = np.asarray(weights[3][emb_idx][0])
            assert position_embeddings.weights[emb_idx].shape == emb_weights.shape, "{} emb does not match".format(position_embeddings[emb_idx])
            position_embeddings.weights[emb_idx] = torch.nn.Parameter(torch.tensor(emb_weights))
    else:
        # otther position encodings
        position_embeddings = np.asarray(weights[3])
        _set_param(torch_model_reformer.embeddings.word_embeddings, torch.tensor(position_embeddings))

    trax_layer_weights = weights[5]
    assert len(torch_model_reformer.encoder.layer) * 4 + 1 == len(trax_layer_weights), "HF and trax model do not have the same number of layers"
    for layer_idx, layer in enumerate(torch_model_reformer.encoder.layer):
        block_weights = trax_layer_weights[4 * layer_idx: 4 * (layer_idx + 1)]
        _set_block_weights_in_torch(block_weights, layer, hidden_size)

    # output weights
    out_weights = weights[6]

    # output layer norm
    layer_norm_out_weight = np.asarray(out_weights[0][0])
    layer_norm_out_bias = np.asarray(out_weights[0][1])
    _set_param(torch_model_reformer.encoder.layer_norm, torch.tensor(layer_norm_out_weight), torch.tensor(layer_norm_out_bias))

    # output embeddings
    output_embed_weights = np.asarray(out_weights[2][0])
    output_embed_bias = np.asarray(out_weights[2][1])
    _set_param(torch_model.lm_head.decoder, torch.tensor(output_embed_weights).transpose(0, 1).contiguous(), torch.tensor(output_embed_bias))


def convert_trax_model_pkl_to_pytorch(model_pkl_path, reformer_config_path, pytorch_dump_folder_path):
    # Initialise PyTorch model
    config = ReformerConfig.from_json_file(reformer_config_path)
    print("Building PyTorch model from configuration: {}".format(str(config)))
    model = ReformerModelWithLMHead(config)

    with GFile(model_pkl_path, 'rb') as f:
        model_weights = pickle.load(f)['weights']

    _set_model_weights_in_torch(model_weights, model, config.hidden_size)
    model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        required=True,
        help="Path to the folder to store the PyTorch model or dataset/vocab.",
    )
    parser.add_argument(
        "--model_pkl_path",
        default="",
        type=str,
        help="An optional path to a TensorFlow checkpoint path to be converted.",
    )
    parser.add_argument(
        "--reformer_config_path",
        default="",
        type=str,
        help="An optional config json file corresponding to the pre-trained BERT model. \n"
        "This specifies the model architecture.",
    )
    args = parser.parse_args()
    convert_trax_model_pkl_to_pytorch(
        args.model_pkl_path,
        args.reformer_config_path,
        args.pytorch_dump_folder_path,
    )
