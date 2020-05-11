# coding=utf-8 # Copyright 2020 Huggingface
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


import unittest

from transformers import is_torch_available

from .test_configuration_common import ConfigTester
from .test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from .utils import require_torch, slow, torch_device

import gin
import numpy as np

# trax imports - to be deleted later
import trax
import jax
from trax.shapes import ShapeDtype as trax_ShapeDtype

if is_torch_available():
    from transformers import (
        ReformerConfig,
        ReformerModel,
        ReformerModelWithLMHead,
        ReformerTokenizer,
        ReformerLayer,
        ReformerAttention,
        REFORMER_PRETRAINED_MODEL_ARCHIVE_MAP,
    )
    import torch


class ReformerModelTester:
    def __init__(
        self,
        parent,
        batch_size=None,
        seq_length=None,
        is_training=None,
        is_decoder=None,
        use_input_mask=None,
        vocab_size=None,
        attention_head_size=None,
        hidden_size=None,
        num_attention_heads=None,
        local_attn_chunk_length=None,
        local_num_chunks_before=None,
        local_num_chunks_after=None,
        num_buckets=None,
        num_hashes=1,
        lsh_attn_chunk_length=None,
        lsh_num_chunks_before=None,
        lsh_num_chunks_after=None,
        chunk_size_lm_head=None,
        chunk_size_feed_forward=None,
        feed_forward_size=None,
        hidden_act=None,
        hidden_dropout_prob=None,
        local_attention_probs_dropout_prob=None,
        lsh_attention_probs_dropout_prob=None,
        max_position_embeddings=None,
        initializer_range=None,
        axial_norm_std=None,
        layer_norm_eps=None,
        axial_pos_embds=None,
        axial_pos_shape=None,
        axial_pos_embds_dim=None,
        attn_layers=None,
        pad_token_id=None,
        eos_token_id=None,
        scope=None,
        hash_seed=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.is_decoder = is_decoder
        self.use_input_mask = use_input_mask
        self.vocab_size = vocab_size
        self.attention_head_size = attention_head_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = len(attn_layers)
        self.local_attn_chunk_length = local_attn_chunk_length
        self.local_num_chunks_after = local_num_chunks_after
        self.local_num_chunks_before = local_num_chunks_before
        self.num_hashes = num_hashes
        self.num_buckets = tuple(num_buckets) if isinstance(num_buckets, list) else num_buckets
        self.lsh_attn_chunk_length = lsh_attn_chunk_length
        self.lsh_num_chunks_after = lsh_num_chunks_after
        self.lsh_num_chunks_before = lsh_num_chunks_before
        self.hidden_act = hidden_act
        self.feed_forward_size = feed_forward_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.local_attention_probs_dropout_prob = local_attention_probs_dropout_prob
        self.lsh_attention_probs_dropout_prob = lsh_attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.axial_pos_embds = axial_pos_embds
        self.axial_pos_shape = tuple(axial_pos_shape)
        self.axial_pos_embds_dim = tuple(axial_pos_embds_dim)
        self.axial_norm_std = axial_norm_std
        self.chunk_size_lm_head = chunk_size_lm_head
        self.chunk_size_feed_forward = chunk_size_feed_forward
        self.scope = scope
        self.attn_layers = attn_layers
        self.pad_token_id = pad_token_id
        self.hash_seed = hash_seed

        attn_chunk_length = local_attn_chunk_length if local_attn_chunk_length is not None else lsh_attn_chunk_length
        num_chunks_after = local_num_chunks_after if local_num_chunks_after is not None else lsh_num_chunks_after
        num_chunks_before = local_num_chunks_before if local_num_chunks_before is not None else lsh_num_chunks_before

        self.encoder_seq_length = seq_length // attn_chunk_length + (self.seq_length % attn_chunk_length != 0)
        self.key_length = (num_chunks_before + num_chunks_after + 1) * attn_chunk_length
        self.chunk_length = attn_chunk_length

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)

        config = ReformerConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            feed_forward_size=self.feed_forward_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            local_attention_probs_dropout_prob=self.local_attention_probs_dropout_prob,
            lsh_attention_probs_dropout_prob=self.lsh_attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            is_decoder=self.is_decoder,
            axial_pos_embds=self.axial_pos_embds,
            axial_pos_shape=self.axial_pos_shape,
            axial_pos_embds_dim=self.axial_pos_embds_dim,
            local_attn_chunk_length=self.local_attn_chunk_length,
            local_num_chunks_after=self.local_num_chunks_after,
            local_num_chunks_before=self.local_num_chunks_before,
            num_hashes=self.num_hashes,
            num_buckets=self.num_buckets,
            lsh_attn_chunk_length=self.lsh_attn_chunk_length,
            lsh_num_chunks_after=self.lsh_num_chunks_after,
            lsh_num_chunks_before=self.lsh_num_chunks_before,
            attn_layers=self.attn_layers,
            pad_token_id=self.pad_token_id,
            hash_seed=self.hash_seed,
        )

        return (
            config,
            input_ids,
            input_mask,
        )

    def check_loss_output(self, result):
        self.parent.assertListEqual(list(result["loss"].size()), [])

    def create_and_check_reformer_model(
        self, config, input_ids, input_mask,
    ):
        model = ReformerModel(config=config)
        model.to(torch_device)
        model.eval()
        (sequence_output,) = model(input_ids, attention_mask=input_mask)
        (sequence_output,) = model(input_ids)

        result = {
            "sequence_output": sequence_output,
        }
        # 2 * hidden_size because we use reversible resnet layers
        self.parent.assertListEqual(
            list(result["sequence_output"].size()), [self.batch_size, self.seq_length, 2 * self.hidden_size],
        )

    def create_and_check_reformer_model_with_lm_backward(
        self, config, input_ids, input_mask,
    ):
        model = ReformerModelWithLMHead(config=config)
        model.to(torch_device)
        model.eval()
        loss = model(input_ids, attention_mask=input_mask, labels=input_ids)[0]
        loss.backward()

    def create_and_check_reformer_with_lm(
        self, config, input_ids, input_mask,
    ):
        model = ReformerModelWithLMHead(config=config)
        model.to(torch_device)
        model.eval()
        loss, prediction_scores = model(input_ids, attention_mask=input_mask, labels=input_ids)
        result = {
            "loss": loss,
            "prediction_scores": prediction_scores,
        }
        self.parent.assertListEqual(
            list(result["prediction_scores"].size()), [self.batch_size, self.seq_length, self.vocab_size],
        )
        self.check_loss_output(result)

    def create_and_check_reformer_model_with_attn_mask(self, config, input_ids, input_mask, is_decoder):
        # no special position embeddings
        config.axial_pos_embds = False
        config.is_decoder = is_decoder

        if self.lsh_attn_chunk_length is not None:
            # need to set chunk length equal sequence length to be certain that chunking works
            config.lsh_attn_chunk_length = self.seq_length

        model = ReformerModel(config=config)
        model.to(torch_device)
        model.eval()
        # set all position encodings to zero so that postions don't matter
        with torch.no_grad():
            embedding = model.embeddings.position_embeddings.embedding
            embedding.weight = torch.nn.Parameter(torch.zeros(embedding.weight.shape).to(torch_device))
            embedding.weight.requires_grad = False

        half_seq_len = self.seq_length // 2
        roll = self.chunk_length

        half_input_ids = input_ids[:, :half_seq_len]

        # normal padded
        attn_mask = torch.cat([torch.ones_like(half_input_ids), torch.zeros_like(half_input_ids)], dim=-1,)
        input_ids_padded = torch.cat(
            [half_input_ids, ids_tensor((self.batch_size, half_seq_len), self.vocab_size)], dim=-1,
        )

        # shifted padded
        input_ids_roll = torch.cat(
            [half_input_ids, ids_tensor((self.batch_size, half_seq_len), self.vocab_size)], dim=-1,
        )
        input_ids_roll = torch.roll(input_ids_roll, roll, dims=-1)
        attn_mask_roll = torch.roll(attn_mask, roll, dims=-1)

        output_padded = model(input_ids_padded, attention_mask=attn_mask)[0][:, :half_seq_len]
        output_padded_rolled = model(input_ids_roll, attention_mask=attn_mask_roll)[0][:, roll : half_seq_len + roll]

        self.parent.assertTrue(torch.allclose(output_padded, output_padded_rolled, atol=1e-3))

    def create_and_check_reformer_layer_dropout_seed(self, config, input_ids, input_mask, is_decoder):
        config.is_decoder = is_decoder
        layer = ReformerLayer(config).to(torch_device)
        layer.train()
        shape = (
            self.batch_size,
            self.seq_length,
            config.hidden_size,
        )  # Batch x SeqLen x hiddenSize

        # get random tensors
        hidden_states = floats_tensor(shape)
        prev_attn_output = floats_tensor(shape)

        # now the random seeds for attention and feed forward is initialized
        # forward tensors with dropout
        layer_outputs = layer(prev_attn_output, hidden_states, attention_mask=input_mask)

        next_attn_output = layer_outputs.attn_output
        next_hidden_states = layer_outputs.hidden_states

        torch.manual_seed(layer.attention_seed)
        attn_outputs = layer.attention(hidden_states, attention_mask=input_mask)
        self.parent.assertTrue(
            torch.allclose(prev_attn_output + attn_outputs.hidden_states, next_attn_output, atol=1e-3,)
        )

        torch.manual_seed(layer.feed_forward_seed)
        feed_forward_hidden_states = layer.feed_forward(next_attn_output)
        self.parent.assertTrue(
            torch.allclose(next_hidden_states, hidden_states + feed_forward_hidden_states, atol=1e-3,)
        )

    def create_and_check_reformer_feed_forward_chunking(self, config, input_ids, input_mask):
        torch.manual_seed(0)
        model = ReformerModel(config=config)
        model.to(torch_device)
        model.eval()
        hidden_states_no_chunk = model(input_ids, attention_mask=input_mask)[0]

        config.chunk_size_lm_head = 1
        config.chunk_size_feed_forward = 1

        torch.manual_seed(0)
        model = ReformerModel(config=config)
        model.to(torch_device)
        model.eval()

        hidden_states_with_chunk = model(input_ids, attention_mask=input_mask)[0]
        self.parent.assertTrue(torch.allclose(hidden_states_no_chunk, hidden_states_with_chunk, atol=1e-3))

    def create_and_check_reformer_feed_backward_chunking(self, config, input_ids, input_mask):
        if not self.is_training:
            return

        # disable dropout
        config.hidden_dropout_prob = 0
        config.local_attention_probs_dropout_prob = 0
        config.lsh_attention_probs_dropout_prob = 0

        torch.manual_seed(0)
        model = ReformerModelWithLMHead(config=config)
        model.to(torch_device)
        model.train()
        model.zero_grad()
        loss_no_chunk, output_no_chunk = model(input_ids, labels=input_ids, attention_mask=input_mask)[:2]
        loss_no_chunk.backward()
        grad_slice_word_no_chunk = model.reformer.embeddings.word_embeddings.weight.grad[0, :5]
        grad_slice_position_factor_1_no_chunk = model.reformer.embeddings.position_embeddings.weights[0][1, 0, -5:]
        grad_slice_position_factor_2_no_chunk = model.reformer.embeddings.position_embeddings.weights[1][0, 1, :5]

        config.chunk_size_lm_head = 1
        config.chunk_size_feed_forward = 1

        torch.manual_seed(0)
        model = ReformerModelWithLMHead(config=config)
        model.to(torch_device)
        model.train()
        model.zero_grad()
        loss_chunk, output_chunk = model(input_ids, labels=input_ids, attention_mask=input_mask)[:2]
        loss_chunk.backward()
        grad_slice_word_chunk = model.reformer.embeddings.word_embeddings.weight.grad[0, :5]
        grad_slice_position_factor_1_chunk = model.reformer.embeddings.position_embeddings.weights[0][1, 0, -5:]
        grad_slice_position_factor_2_chunk = model.reformer.embeddings.position_embeddings.weights[1][0, 1, :5]
        self.parent.assertTrue(torch.allclose(loss_chunk, loss_no_chunk, atol=1e-3))
        self.parent.assertTrue(torch.allclose(grad_slice_word_no_chunk, grad_slice_word_chunk, atol=1e-3))
        self.parent.assertTrue(
            torch.allclose(grad_slice_position_factor_1_chunk, grad_slice_position_factor_1_no_chunk, atol=1e-3)
        )
        self.parent.assertTrue(
            torch.allclose(grad_slice_position_factor_2_chunk, grad_slice_position_factor_2_no_chunk, atol=1e-3)
        )

    def create_and_check_reformer_random_seed(self, config, input_ids, input_mask):
        layer = ReformerLayer(config).to(torch_device)
        layer.train()

        shape = (
            self.batch_size,
            self.seq_length,
            config.hidden_size,
        )  # Batch x SeqLen x hiddenSize

        hidden_states = floats_tensor(shape)
        attn_output = floats_tensor(shape)

        seeds = []
        for _ in range(100):
            layer_outputs = layer(attn_output, hidden_states, attention_mask=input_mask)
            attn_output = layer_outputs.attn_output
            hidden_states = layer_outputs.hidden_states
            torch.manual_seed(layer.attention_seed)
            seeds.append(layer.attention_seed)
        self.parent.assertGreater(len(set(seeds)), 70)

        seeds = []
        for _ in range(100):
            layer_outputs = layer(attn_output, hidden_states, attention_mask=input_mask)
            attn_output = layer_outputs.attn_output
            hidden_states = layer_outputs.hidden_states
            torch.manual_seed(layer.feed_forward_seed)
            seeds.append(layer.feed_forward_seed)
        self.parent.assertGreater(len(set(seeds)), 70)

    def create_and_check_reformer_model_fp16_forward(self, config, input_ids, input_mask):
        model = ReformerModel(config=config)
        model.to(torch_device)
        model.half()
        model.eval()
        output = model(input_ids, attention_mask=input_mask)[0]
        self.parent.assertFalse(torch.isnan(output).any().item())

    def create_and_check_reformer_model_fp16_generate(self, config, input_ids, input_mask):
        model = ReformerModelWithLMHead(config=config)
        model.to(torch_device)
        model.half()
        model.eval()
        output = model.generate(input_ids, attention_mask=input_mask, do_sample=False)
        self.parent.assertFalse(torch.isnan(output).any().item())

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (config, input_ids, input_mask,) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


class ReformerTesterMixin:
    """
        Reformer Local and Reformer LSH run essentially the same tests
    """

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_reformer_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_reformer_model(*config_and_inputs)

    def test_reformer_lm_model_backward(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_reformer_model_with_lm_backward(*config_and_inputs)

    def test_reformer_model_attn_masking(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_reformer_model_with_attn_mask(*config_and_inputs, True)
        self.model_tester.create_and_check_reformer_model_with_attn_mask(*config_and_inputs, False)

    def test_reformer_with_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_reformer_with_lm(*config_and_inputs)

    def test_reformer_layer_training_dropout(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_reformer_layer_dropout_seed(*config_and_inputs, True)
        self.model_tester.create_and_check_reformer_layer_dropout_seed(*config_and_inputs, False)

    def test_reformer_chunking_forward_equality(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_reformer_feed_forward_chunking(*config_and_inputs)

    def test_reformer_chunking_backward_equality(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_reformer_feed_backward_chunking(*config_and_inputs)

    @slow
    def test_dropout_random_seed_is_changing(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_reformer_random_seed(*config_and_inputs)

    @unittest.skipIf(torch_device == "cpu", "Cant do half precision")
    def test_reformer_model_fp16_forward(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_reformer_model_fp16_forward(*config_and_inputs)

    @unittest.skipIf(torch_device == "cpu", "Cant do half precision")
    def test_reformer_model_fp16_generate(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_reformer_model_fp16_generate(*config_and_inputs)


@require_torch
class ReformerLocalAttnModelTest(ModelTesterMixin, ReformerTesterMixin, unittest.TestCase):
    all_model_classes = (ReformerModel, ReformerModelWithLMHead) if is_torch_available() else ()
    all_generative_model_classes = (ReformerModelWithLMHead,) if is_torch_available() else ()
    test_pruning = False
    test_headmasking = False
    test_torchscript = False

    def prepare_kwargs(self):
        return {
            "batch_size": 13,
            "seq_length": 32,
            "is_training": True,
            "is_decoder": False,
            "use_input_mask": True,
            "vocab_size": 32,
            "attention_head_size": 16,
            "hidden_size": 32,
            "num_attention_heads": 2,
            "local_attn_chunk_length": 4,
            "local_num_chunks_before": 1,
            "local_num_chunks_after": 0,
            "chunk_size_lm_head": 0,
            "chunk_size_feed_forward": 0,
            "feed_forward_size": 32,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "local_attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "initializer_range": 0.02,
            "axial_norm_std": 1.0,
            "layer_norm_eps": 1e-12,
            "axial_pos_embds": True,
            "axial_pos_shape": [4, 8],
            "axial_pos_embds_dim": [16, 16],
            "attn_layers": ["local", "local", "local", "local"],
            "pad_token_id": 0,
            "eos_token_id": 2,
            "scope": None,
            "hash_seed": 0,
        }

    def setUp(self):
        tester_kwargs = self.prepare_kwargs()
        self.model_tester = ReformerModelTester(self, **tester_kwargs)
        self.config_tester = ConfigTester(self, config_class=ReformerConfig, hidden_size=37)

    @slow
    def test_model_from_pretrained(self):
        for model_name in list(REFORMER_PRETRAINED_MODEL_ARCHIVE_MAP.keys())[:1]:
            model = ReformerModelWithLMHead.from_pretrained(model_name)
            self.assertIsNotNone(model)


@require_torch
class ReformerLSHAttnModelTest(ModelTesterMixin, unittest.TestCase, ReformerTesterMixin):
    all_model_classes = (ReformerModel, ReformerModelWithLMHead) if is_torch_available() else ()
    all_generative_model_classes = (ReformerModelWithLMHead,) if is_torch_available() else ()
    test_pruning = False
    test_headmasking = False
    test_torchscript = False

    def prepare_kwargs(self):
        return {
            "batch_size": 13,
            "seq_length": 13,
            "use_input_mask": True,
            "is_training": False,
            "is_decoder": False,
            "vocab_size": 32,
            "attention_head_size": 16,
            "hidden_size": 64,
            "num_attention_heads": 2,
            "num_buckets": 2,
            "num_hashes": 4,
            "lsh_attn_chunk_length": 4,
            "lsh_num_chunks_before": 2,
            "lsh_num_chunks_after": 3,
            "chunk_size_lm_head": 5,
            "chunk_size_feed_forward": 6,
            "feed_forward_size": 32,
            "hidden_act": "relu",
            "hidden_dropout_prob": 0.1,
            "lsh_attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "initializer_range": 0.02,
            "axial_norm_std": 1.0,
            "layer_norm_eps": 1e-12,
            "axial_pos_embds": True,
            "axial_pos_shape": [4, 8],
            "axial_pos_embds_dim": [16, 48],
            "attn_layers": ["lsh", "lsh", "lsh", "lsh"],
            "pad_token_id": 0,
            "eos_token_id": 2,
            "scope": None,
            "hash_seed": 0,
        }

    def setUp(self):
        tester_kwargs = self.prepare_kwargs()
        self.model_tester = ReformerModelTester(self, **tester_kwargs)
        self.config_tester = ConfigTester(self, config_class=ReformerConfig, hidden_size=37)


@require_torch
class ReformerIntegrationTests(unittest.TestCase):
    """
    These integration tests test the current layer activations and gradients againts the output of the Hugging Face Reformer model at time of integration: 29/04/2020. During integration, the model was tested against the output of the official Trax ReformerLM model for various cases ("lsh" only, "local" only, masked / non-masked, different chunk length, ....). In order to recover the original trax integration tests, one should use patrickvonplaten's fork of trax and the code that lives on the branch `branch_to_save_trax_integration_tests`.
    """

    def _get_basic_config_and_input(self):
        config = {
            "vocab_size": 320,
            "attention_head_size": 8,
            "hidden_size": 16,
            "num_attention_heads": 2,
            "num_buckets": 2,
            "num_hashes": 4,
            "lsh_attn_chunk_length": 4,
            "local_attn_chunk_length": 4,
            "lsh_num_chunks_before": 1,
            "lsh_num_chunks_after": 0,
            "local_num_chunks_before": 1,
            "local_num_chunks_after": 0,
            "chunk_size_lm_head": 0,
            "chunk_size_feed_forward": 0,
            "feed_forward_size": 32,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.0,
            "lsh_attention_probs_dropout_prob": 0.0,
            "local_attention_probs_dropout_prob": 0.0,
            "max_position_embeddings": 32,
            "initializer_range": 0.02,
            "axial_norm_std": 1.0,
            "layer_norm_eps": 1e-12,
            "sinusoidal_pos_embds": False,
            "axial_pos_embds": True,
            "axial_pos_shape": [4, 8],
            "axial_pos_embds_dim": [8, 8],
            "hash_seed": 0,
            "is_decoder": True,
        }
        return config

    def _get_hidden_states(self):
        return torch.tensor(
            [
                [
                    [
                        1.90826353e00,
                        -1.45999730e00,
                        -6.20405462e-01,
                        1.52503433e00,
                        -3.64464232e-01,
                        -8.27359235e-01,
                        8.39670803e-01,
                        2.44492178e-01,
                        4.98332758e-01,
                        2.69175139e00,
                        -7.08081422e-03,
                        1.04915401e00,
                        -1.83476661e00,
                        7.67220476e-01,
                        2.98580543e-01,
                        2.84803992e-02,
                    ],
                    [
                        -2.66374286e-02,
                        4.33497576e-01,
                        3.10386309e-01,
                        5.46039944e-01,
                        -2.47292666e-04,
                        -7.52305019e-01,
                        2.39162103e-01,
                        7.25216186e-01,
                        -7.58357372e-01,
                        4.20635998e-01,
                        -4.04739919e-02,
                        1.59924145e-01,
                        2.05135748e00,
                        -1.15997978e00,
                        5.37166397e-01,
                        2.62873606e-01,
                    ],
                    [
                        1.85247482e-01,
                        7.07046037e-01,
                        -6.77089715e-01,
                        -2.24209655e00,
                        -3.75307980e-02,
                        -8.59380874e-01,
                        -2.81027884e00,
                        1.01276376e00,
                        -1.69438001e00,
                        4.17574660e-01,
                        -1.49196962e00,
                        -1.76483717e00,
                        -1.94566312e-01,
                        -1.71183858e00,
                        7.72903565e-01,
                        -1.11557056e00,
                    ],
                    [
                        9.46069193e-01,
                        1.53417623e-01,
                        -9.58686996e-01,
                        1.18126669e-01,
                        1.75967724e00,
                        1.62194590e00,
                        -5.74108159e-01,
                        6.79920443e-01,
                        5.44028163e-01,
                        2.05466114e-01,
                        -3.63045868e-01,
                        2.41865062e-01,
                        3.20348382e-01,
                        -9.05611176e-01,
                        -1.92690727e-01,
                        -1.19917547e00,
                    ],
                ]
            ],
            dtype=torch.float32,
            device=torch_device,
        )

    def _get_attn_mask(self):
        return torch.tensor([[0, 1, 0, 0]], dtype=torch.long, device=torch_device)

    def _get_input_ids_and_mask(self):
        mask = torch.tensor(
            [
                [1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0],
            ],
            dtype=torch.long,
            device=torch_device,
        )

        input_ids = torch.tensor(
            [
                [
                    89,
                    279,
                    286,
                    84,
                    194,
                    316,
                    182,
                    28,
                    283,
                    37,
                    169,
                    7,
                    253,
                    267,
                    107,
                    250,
                    44,
                    7,
                    102,
                    62,
                    3,
                    243,
                    171,
                    265,
                    302,
                    48,
                    164,
                    264,
                    148,
                    229,
                    280,
                    150,
                ],
                [
                    9,
                    192,
                    66,
                    112,
                    163,
                    83,
                    135,
                    70,
                    224,
                    96,
                    31,
                    80,
                    196,
                    80,
                    63,
                    22,
                    85,
                    100,
                    47,
                    283,
                    0,
                    163,
                    126,
                    143,
                    195,
                    82,
                    53,
                    82,
                    18,
                    27,
                    182,
                    52,
                ],
            ],
            dtype=torch.long,
            device=torch_device,
        )

        return input_ids, mask

    def test_lsh_layer_forward(self):
        config = self._get_basic_config_and_input()
        config["attn_layers"] = ["lsh"]
        config["is_decoder"] = False
        hidden_states = self._get_hidden_states()
        torch.manual_seed(0)
        layer = ReformerLayer(ReformerConfig(**config)).to(torch_device)
        layer.eval()
        reformer_output = layer(prev_attn_output=hidden_states.clone(), hidden_states=hidden_states)
        output_slice = reformer_output.hidden_states[0, 0, :5]
        expected_output_slice = torch.tensor(
            [1.6879, -1.3083, -0.4708, 1.3555, -0.6292], dtype=torch.float, device=torch_device,
        )
        self.assertTrue(torch.allclose(output_slice, expected_output_slice, atol=1e-3))

    def test_lsh_layer_forward_complex(self):
        config = self._get_basic_config_and_input()
        config["attn_layers"] = ["lsh"]
        config["num_buckets"] = [2, 4]
        attn_mask = self._get_attn_mask()
        hidden_states = self._get_hidden_states()
        torch.manual_seed(0)
        layer = ReformerLayer(ReformerConfig(**config)).to(torch_device)
        layer.eval()
        reformer_output = layer(
            prev_attn_output=hidden_states.clone(), hidden_states=hidden_states, attention_mask=attn_mask,
        )
        output_slice = reformer_output.hidden_states[0, 0, :5]
        expected_output_slice = torch.tensor(
            [1.6439, -1.2306, -0.5108, 1.3006, -0.6537], dtype=torch.float, device=torch_device,
        )
        self.assertTrue(torch.allclose(output_slice, expected_output_slice, atol=1e-3))

    def test_local_layer_forward(self):
        config = self._get_basic_config_and_input()
        config["attn_layers"] = ["local"]
        config["is_decoder"] = False
        hidden_states = self._get_hidden_states()
        torch.manual_seed(0)
        layer = ReformerLayer(ReformerConfig(**config)).to(torch_device)
        layer.eval()
        reformer_output = layer(prev_attn_output=hidden_states, hidden_states=hidden_states)
        output_slice = reformer_output.hidden_states[0, 0, :5]
        expected_output_slice = torch.tensor(
            [1.4212, -2.0576, -0.9688, 1.4599, -0.1344], dtype=torch.float, device=torch_device,
        )
        self.assertTrue(torch.allclose(output_slice, expected_output_slice, atol=1e-3))

    def test_local_layer_forward_complex(self):
        config = self._get_basic_config_and_input()
        config["attn_layers"] = ["local"]
        attn_mask = self._get_attn_mask()
        hidden_states = self._get_hidden_states()
        torch.manual_seed(0)
        layer = ReformerLayer(ReformerConfig(**config)).to(torch_device)
        layer.eval()
        reformer_output = layer(prev_attn_output=hidden_states, hidden_states=hidden_states, attention_mask=attn_mask,)
        output_slice = reformer_output.hidden_states[0, 0, :5]
        expected_output_slice = torch.tensor(
            [1.5476, -1.9020, -0.9902, 1.5013, -0.1950], dtype=torch.float, device=torch_device,
        )
        self.assertTrue(torch.allclose(output_slice, expected_output_slice, atol=1e-3))

    def test_lsh_model_forward(self):
        config = self._get_basic_config_and_input()
        config["attn_layers"] = ["lsh", "lsh", "lsh", "lsh"]
        config["num_buckets"] = [2, 4]
        torch.manual_seed(0)
        model = ReformerModel(ReformerConfig(**config)).to(torch_device)
        model.eval()
        input_ids, attn_mask = self._get_input_ids_and_mask()
        hidden_states = model(input_ids=input_ids, attention_mask=attn_mask)[0]
        output_slice = hidden_states[0, 0, :5]
        expected_output_slice = torch.tensor(
            [-0.9896, -0.9396, -1.0831, -0.0597, 0.2456], dtype=torch.float, device=torch_device,
        )
        self.assertTrue(torch.allclose(output_slice, expected_output_slice, atol=1e-3))

    def test_local_model_forward(self):
        config = self._get_basic_config_and_input()
        config["attn_layers"] = ["local", "local", "local", "local"]
        torch.manual_seed(0)
        model = ReformerModel(ReformerConfig(**config)).to(torch_device)
        model.eval()
        input_ids, attn_mask = self._get_input_ids_and_mask()
        hidden_states = model(input_ids=input_ids, attention_mask=attn_mask)[0]
        output_slice = hidden_states[0, 0, :5]
        expected_output_slice = torch.tensor(
            [-1.6791, 0.7171, 0.1594, 0.4063, 1.2584], dtype=torch.float, device=torch_device,
        )
        self.assertTrue(torch.allclose(output_slice, expected_output_slice, atol=1e-3))

    def test_lm_model_forward(self):
        config = self._get_basic_config_and_input()
        config["attn_layers"] = ["local", "lsh", "local", "lsh", "local", "lsh"]
        config["num_buckets"] = [2, 4]
        config["is_decoder"] = False
        torch.manual_seed(0)
        model = ReformerModelWithLMHead(ReformerConfig(**config)).to(torch_device)
        model.eval()
        input_ids, attn_mask = self._get_input_ids_and_mask()
        hidden_states = model(input_ids=input_ids, attention_mask=attn_mask)[0]
        output_slice = hidden_states[1, -1, :5]
        expected_output_slice = torch.tensor(
            [0.0324, -0.0121, 0.0615, 0.0031, -0.0297], dtype=torch.float, device=torch_device,
        )
        self.assertTrue(torch.allclose(output_slice, expected_output_slice, atol=1e-3))

    def test_local_lm_model_grad(self):
        config = self._get_basic_config_and_input()
        config["attn_layers"] = ["local", "local", "local", "local"]
        config["hidden_dropout_prob"] = 0.0
        config["local_attention_probs_dropout_prob"] = 0.0
        torch.manual_seed(0)
        model = ReformerModelWithLMHead(ReformerConfig(**config)).to(torch_device)
        model.train()
        model.zero_grad()
        input_ids, _ = self._get_input_ids_and_mask()
        loss = model(input_ids=input_ids, labels=input_ids)[0]

        self.assertTrue(torch.allclose(loss, torch.tensor(5.7786, dtype=torch.float, device=torch_device), atol=1e-3))
        loss.backward()

        # check last grads to cover all proable errors
        grad_slice_word = model.reformer.embeddings.word_embeddings.weight.grad[0, :5]
        expected_grad_slice_word = torch.tensor(
            [-0.0005, 0.0001, 0.0002, 0.0003, 0.0006], dtype=torch.float, device=torch_device,
        )
        grad_slice_position_factor_1 = model.reformer.embeddings.position_embeddings.weights[0][1, 0, -5:]
        expected_grad_slice_pos_fac_1 = torch.tensor(
            [0.0037, -1.3793, -1.0231, -1.5230, -2.5306], dtype=torch.float, device=torch_device,
        )
        grad_slice_position_factor_2 = model.reformer.embeddings.position_embeddings.weights[1][0, 1, :5]
        expected_grad_slice_pos_fac_2 = torch.tensor(
            [-1.3165, 0.5168, 0.7785, 1.0811, -0.9830], dtype=torch.float, device=torch_device,
        )
        self.assertTrue(torch.allclose(grad_slice_word, expected_grad_slice_word, atol=1e-3))
        self.assertTrue(torch.allclose(grad_slice_position_factor_1, expected_grad_slice_pos_fac_1, atol=1e-3))
        self.assertTrue(torch.allclose(grad_slice_position_factor_2, expected_grad_slice_pos_fac_2, atol=1e-3))

    def test_lsh_lm_model_grad(self):
        config = self._get_basic_config_and_input()
        config["attn_layers"] = ["lsh", "lsh", "lsh", "lsh"]
        config["hidden_dropout_prob"] = 0.0
        config["lsh_attention_probs_dropout_prob"] = 0.0
        config["num_buckets"] = [2, 4]
        config["num_hashes"] = 6
        torch.manual_seed(0)
        model = ReformerModelWithLMHead(ReformerConfig(**config)).to(torch_device)
        model.train()
        model.zero_grad()
        input_ids, _ = self._get_input_ids_and_mask()
        loss = model(input_ids=input_ids, labels=input_ids)[0]

        self.assertTrue(torch.allclose(loss, torch.tensor(5.7819, dtype=torch.float, device=torch_device), atol=1e-3))
        loss.backward()
        # check last grads to cover all proable errors
        grad_slice_word = model.reformer.embeddings.word_embeddings.weight.grad[0, :5]
        expected_grad_slice_word = torch.tensor(
            [2.6357e-05, 4.3358e-04, -8.4985e-04, 1.0094e-04, 3.8954e-04], dtype=torch.float, device=torch_device,
        )
        grad_slice_position_factor_1 = model.reformer.embeddings.position_embeddings.weights[0][1, 0, -5:]
        expected_grad_slice_pos_fac_1 = torch.tensor(
            [-0.0984, 0.6283, 0.4282, 1.2960, 0.6897], dtype=torch.float, device=torch_device,
        )
        grad_slice_position_factor_2 = model.reformer.embeddings.position_embeddings.weights[1][0, 1, :5]
        expected_grad_slice_pos_fac_2 = torch.tensor(
            [0.4626, -0.0231, -0.0172, 0.1081, 0.3805], dtype=torch.float, device=torch_device,
        )
        self.assertTrue(torch.allclose(grad_slice_word, expected_grad_slice_word, atol=1e-3))
        self.assertTrue(torch.allclose(grad_slice_position_factor_1, expected_grad_slice_pos_fac_1, atol=1e-3))
        self.assertTrue(torch.allclose(grad_slice_position_factor_2, expected_grad_slice_pos_fac_2, atol=1e-3))

    @slow
    def test_pretrained_generate_crime_and_punish(self):
        model = ReformerModelWithLMHead.from_pretrained("google/reformer-crime-and-punishment").to(torch_device)
        tokenizer = ReformerTokenizer.from_pretrained("google/reformer-crime-and-punishment")
        model.eval()

        input_ids = tokenizer.encode("A few months later", return_tensors="pt").to(torch_device)
        output_ids = model.generate(
            input_ids, max_length=50, num_beams=4, early_stopping=True, do_sample=False, num_hashes=8
        )
        output_text = tokenizer.decode(output_ids[0])
        self.assertEqual(
            output_text,
            "A few months later state expression in his ideas, at the first entrance. He was positively for an inst",
        )


@require_torch
class ReformerIntegrationTestsDynamic(unittest.TestCase):
    # This code has to be used with patrickvonplaten's fork of trax to work
    def test_lsh_layer(self):
        config = ReformerConfig(hash_seed=0)
        shape = (3, 64, config.hidden_size)  # Batch x SeqLen x hiddenSize
        np_input = np.random.rand(*shape)

        trax_layer = self.load_lsh_layer(config)
        input_signature = trax_ShapeDtype(shape, np.float32)
        trax_weights, trax_state = trax_layer.init(input_signature)

        mask = np.ones(shape[:-1], dtype=np.int32)

        trax_output = trax_layer(np_input, weights=trax_weights, state=trax_state)

        trax_torch_output = torch.tensor(np.asarray(trax_output))

        hf_input = torch.tensor(np_input, dtype=torch.float)
        config.attn_layers = ["lsh"]
        hf_layer = ReformerAttention(config)
        self._set_layer_weights_in_torch_lsh(trax_weights, hf_layer, config.hidden_size)
        hf_layer.eval()

        hf_attention_all_heads = hf_layer.self_attention(
            hf_input, attention_mask=torch.tensor(mask)
        )[0]
        hf_output = hf_layer.output(hf_attention_all_heads)

        self.assertTrue(torch.allclose(hf_output, trax_torch_output, atol=1e-3))

    def test_local_layer(self):
        config = ReformerConfig(hash_seed=0)
        shape = (1, 64, config.hidden_size)  # Batch x SeqLen x hiddenSize
        np_input = np.random.rand(*shape)

        trax_layer = self.load_local_layer(config)
        input_signature = trax_ShapeDtype(shape, np.float32)
        trax_weights, trax_state = trax_layer.init(input_signature)
        mask = np.ones(shape[:-1], dtype=np.int32)

        trax_output = trax_layer(np_input, weights=trax_weights, state=trax_state)

        hf_input = torch.tensor(np_input, dtype=torch.float)
        config.attn_layers = ["local"]
        hf_layer = ReformerAttention(config)
        self._set_layer_weights_in_torch_local(
            trax_weights, hf_layer, config.hidden_size
        )
        hf_layer.eval()

        hf_attention_all_heads = hf_layer.self_attention(
            hf_input, attention_mask=torch.tensor(mask)
        )[0]
        hf_output = hf_layer.output(hf_attention_all_heads)

        trax_torch_output = torch.tensor(np.asarray(trax_output))
        self.assertTrue(torch.allclose(hf_output, trax_torch_output, atol=1e-3))

    def test_reformer_lm_model(self):
        config = ReformerConfig(
                axial_pos_embds=True,
                hash_seed=0,
                is_decoder=True,
                axial_pos_shape=[32, 16],
                axial_pos_embds_dim=[64, 192],
                attn_layers=["local", "local", "local", "local"],
                local_attention_probs_dropout_prob=0.0,
                lsh_attention_probs_dropout_prob=0.0,
                hidden_dropout_prob=0.0,
        )

        shape = (1, 512)  # Batch x SeqLen x ModelDimPerHead

        np_input = np.random.randint(0, config.vocab_size, size=shape)
        np_input_2 = np.asarray(np_input, np.float32)
        mask = np.ones_like(np_input, dtype=np.float32)
#        mask[0, 10:20] = 0
        np_zeros = np.zeros((shape[0], 1), dtype=np.int)

        # choose one of the following two. "train" tests gradients. "test" tests forward only.
        mode = "train"
#        mode = "eval"

        trax_model = self.load_reformer_lm_model(config, mode=mode)

        assert (
            config.is_decoder is True
        ), "trax can only test casaul mask for ReformerLM. Use tests for layers to test non-casaul mask"

        if mode == "train":
            trax_model = trax.layers.Serial(trax_model, trax.layers.CrossEntropyLoss())
            input_signature = (
                trax_ShapeDtype(shape, np.int32),
                trax_ShapeDtype(shape, np.float32),
                trax_ShapeDtype(shape, np.float32),
            )
            trax_weights, trax_state = trax_model.init(input_signature)
            trax_input = (np_input, np_input_2, mask)
            torch_trax_weights = trax_weights[0]
        else:
            input_signature = trax_ShapeDtype(shape, np.int32)
            trax_weights, trax_state = trax_model.init(input_signature)
            trax_input = np_input
            trax_output = trax_model(trax_input, weights=trax_weights, state=trax_state)
            trax_torch_output = torch.tensor(np.asarray(trax_output[0]))
            torch_trax_weights = trax_weights

        if mode != "predict":
            hf_input = torch.cat(
                [torch.tensor(np_zeros), torch.tensor(np_input[:, :-1])], dim=-1
            )
            attention_mask = torch.tensor(mask)
            hf_labels = (
                -100 * (1 - attention_mask) + torch.tensor(np_input) * attention_mask
            ).to(dtype=torch.long)
        else:
            hf_input = torch.tensor(np_input)

        hf_model = ReformerModelWithLMHead(config)
        self._set_model_weights_in_torch(
            torch_trax_weights, hf_model, config.hidden_size
        )

        if mode == "train":
            # uncomment line to fix hf_input_shifting in ReformerWithLMHead
            hf_model.train()
            # Trax does not really use attention masks in their layers in this setup. The just
            # mask the final loss
            loss = hf_model(hf_input, labels=hf_labels)[0]
        else:
            hf_output = hf_model(hf_input)
            hf_output = torch.nn.functional.log_softmax(hf_output[0], dim=-1)
            self.assertTrue(torch.allclose(hf_output, trax_torch_output, atol=1e-4))

        if mode == "train":
            hf_model.zero_grad()
            loss.backward()

            def model_and_loss_call(weights, batch, state):
                res = trax_model(batch, weights=weights, state=state)
                return res, trax_model.state

            grad_fn = jax.grad(model_and_loss_call, has_aux=True)
            grads, state = grad_fn(trax_weights, trax_input, trax_state)

            all_test_correct = self._set_model_weights_in_torch(
                grads[0], hf_model, config.hidden_size, set_params=False
            )
            self.assertTrue(all_test_correct)

    def test_backprop_lm_model(self):
        config = ReformerConfig()

        shape = (1, 192)  # Batch x SeqLen x ModelDimPerHead
        input_ids = torch.tensor(
            np.random.randint(0, config.vocab_size, size=shape),
            dtype=torch.long,
            device=torch_device,
        )

        model = ReformerModelWithLMHead(config)
        loss = model(input_ids, labels=input_ids)[0]
        loss.backward()

    # use github old branch to make this test work. Pretrained weights
    # cannot be loaded into new code anymore
    def test_pretrained_crime_and_punishment_lm_model(self):
        hf_model = ReformerModelWithLMHead.from_pretrained(
            "google/reformer-crime-and-punishment"
        )
        config = hf_model.config

        trax_model_path = (
            "/home/patrick/hugging_face/models/trained_reformer_colab/model.pkl"
        )

        shape = (1, 512)
        np_input = np.random.randint(0, config.vocab_size, size=shape)

        hf_input = torch.tensor(np_input)

        input_signature = trax_ShapeDtype(shape, np.int32)
        trax_model = self.load_crime_and_punishment_model(
            trax_model_path, input_signature
        )

        hf_output = hf_model(hf_input)
        log_softmax_output = torch.nn.functional.log_softmax(hf_output[0], dim=-1)

        trax_output = trax_model(np_input)
        trax_torch_output = torch.tensor(np.asarray(trax_output[0]))

        self.assertTrue(
            torch.allclose(log_softmax_output, trax_torch_output, atol=1e-3)
        )

    def test_pretrained_enwiki8_lm_model(self):
        hf_model = ReformerModelWithLMHead.from_pretrained(
            "google/reformer-crime-and-punishment"
        )
        config = hf_model.config

        trax_model_path = (
            "/home/patrick/hugging_face/reformer/enwik8_model/reformer_enwik8_model.pkl"
        )

        shape = (1, 512)
        np_input = np.random.randint(0, config.vocab_size, size=shape)

        hf_input = torch.tensor(np_input)

        input_signature = trax_ShapeDtype(shape, np.int32)
        trax_model = self.load_enwik8_model(
            trax_model_path, input_signature
        )

        hf_output = hf_model(hf_input)
        log_softmax_output = torch.nn.functional.log_softmax(hf_output[0], dim=-1)

        trax_output = trax_model(np_input)
        trax_torch_output = torch.tensor(np.asarray(trax_output[0]))

        self.assertTrue(
            torch.allclose(log_softmax_output, trax_torch_output, atol=1e-3)
        )

    def load_lsh_layer(self, config, mode="eval"):
        gin_config = """
            import trax.layers
            # Parameters for LSHSelfAttention:
            # ==============================================================================
            LSHSelfAttention.n_heads = {}
            LSHSelfAttention.d_qk = {}
            LSHSelfAttention.d_v = {}
            LSHSelfAttention.chunk_len = {}
            LSHSelfAttention.n_chunks_before = {}
            LSHSelfAttention.n_chunks_after = {}
            LSHSelfAttention.n_hashes = {}
            LSHSelfAttention.n_buckets = {}
            LSHSelfAttention.attention_dropout = {}
            LSHSelfAttention.output_dropout = {}
            LSHSelfAttention.lsh_seed = {}
            LSHSelfAttention.causal= {}
            LSHSelfAttention.use_reference_code = True
            """.format(
            config.num_attention_heads,
            config.attention_head_size,
            config.attention_head_size,
            config.lsh_attn_chunk_length,
            config.lsh_num_chunks_before,
            config.lsh_num_chunks_after,
            config.num_hashes,
            config.num_buckets,
            config.lsh_attention_probs_dropout_prob,
            config.hidden_dropout_prob,
            config.hash_seed,
            config.is_decoder,
        )
        gin.parse_config(gin_config)
        layer = trax.layers.LSHSelfAttention(mode=mode)
        return layer

    def load_local_layer(self, config, mask=False, mode="eval"):
        gin_config = """
            import trax.layers
            # Parameters for SelfAttention:
            # ==============================================================================
            SelfAttention.n_heads = {}
            SelfAttention.d_qk = {}
            SelfAttention.d_v = {}
            SelfAttention.chunk_len = {}
            SelfAttention.n_chunks_before = {}
            SelfAttention.n_chunks_after = {}
            SelfAttention.attention_dropout = {}
            SelfAttention.output_dropout = {}
            SelfAttention.causal = {}
            SelfAttention.masked= {}
            SelfAttention.use_reference_code = True
            """.format(
            config.num_attention_heads,
            config.attention_head_size,
            config.attention_head_size,
            config.local_attn_chunk_length,
            config.local_num_chunks_before,
            config.local_num_chunks_after,
            config.local_attention_probs_dropout_prob,
            config.hidden_dropout_prob,
            config.is_decoder,
            mask,
        )
        gin.parse_config(gin_config)
        layer = trax.layers.SelfAttention(mode=mode)
        return layer

    def load_reformer_lm_model(self, config, mode="eval"):
        if config.hidden_act == "gelu":
            hidden_act = "Gelu"
        elif config.hidden_act == "relu":
            hidden_act = "Relu"
        else:
            raise ValueError()
        attn_type = config.attn_layers[0]
        if attn_type == "lsh":
            attn_type = "LSHSelfAttention"
        elif attn_type == "local":
            attn_type = "SelfAttention"
        else:
            raise ValueError()

        axial_pos_shape = config.axial_pos_shape
        d_axial_pos_embs = config.axial_pos_embds_dim

        gin_config = """
            import trax.layers
            import trax.models
            # Parameters for LSHSelfAttention:
            # ==============================================================================
            LSHSelfAttention.chunk_len = {}
            LSHSelfAttention.predict_mem_len = {}
            LSHSelfAttention.predict_drop_len = {}
            LSHSelfAttention.n_chunks_before = {}
            LSHSelfAttention.n_chunks_after = {}
            LSHSelfAttention.n_hashes = {}
            LSHSelfAttention.n_buckets = {}
            LSHSelfAttention.lsh_seed = {}
            LSHSelfAttention.causal= {}
            LSHSelfAttention.use_reference_code = True
            # Parameters for SelfAttention:
            # ==============================================================================
            SelfAttention.chunk_len = {}
            SelfAttention.n_chunks_before = {}
            SelfAttention.n_chunks_after = {}
            SelfAttention.causal= {}
            SelfAttention.use_reference_code = True
            SelfAttention.share_qk = False
            # Parameters for ReformerLM:
            # ==============================================================================
            ReformerLM.vocab_size = {}
            ReformerLM.d_model = {}
            ReformerLM.d_ff = {}
            ReformerLM.d_attention_key = {}
            ReformerLM.d_attention_value = {}
            ReformerLM.n_layers = {}
            ReformerLM.n_heads = {}
            ReformerLM.max_len = {}
            ReformerLM.axial_pos_shape = {}
            ReformerLM.d_axial_pos_embs = {}
            ReformerLM.ff_chunk_size = {}
            ReformerLM.ff_activation = @trax.layers.{}
            ReformerLM.attention_type = @trax.layers.{}
            ReformerLM.dropout = 0.0
            ReformerLM.ff_use_sru = 0
            """.format(
            config.lsh_attn_chunk_length,
            config.lsh_attn_chunk_length,
            config.lsh_attn_chunk_length // 2,
            config.lsh_num_chunks_before,
            config.lsh_num_chunks_after,
            config.num_hashes,
            config.num_buckets,
            config.hash_seed,
            config.is_decoder,
            config.local_attn_chunk_length,
            config.local_num_chunks_before,
            config.local_num_chunks_after,
            config.is_decoder,
            config.vocab_size,
            config.hidden_size,
            config.feed_forward_size,
            config.attention_head_size,
            config.attention_head_size,
            config.num_hidden_layers,
            config.num_attention_heads,
            config.max_position_embeddings,
            axial_pos_shape,
            d_axial_pos_embs,
            config.chunk_size_feed_forward,
            hidden_act,
            attn_type,
        )
        gin.parse_config(gin_config)
        model = trax.models.ReformerLM(mode=mode)
        return model

    # (PVP) - delete when enwiki works
    def load_crime_and_punishment_model(
        self, trax_model_path, input_signature, mode="predict"
    ):
        gin.parse_config(
            """
            import trax.layers
            import trax.models
            import trax.optimizers
            import trax.supervised.inputs
            import trax.supervised.trainer_lib
            # Parameters that will vary between experiments:
            # ==============================================================================
            train.model = @trax.models.ReformerLM
            # Our model will have 6 layers, alternating between the LSH attention proposed
            # in the Reformer paper and local attention within a certain context window.
            n_layers = 6
            attn_type = [
              @SelfAttention,
              @LSHSelfAttention,
              @SelfAttention,
              @LSHSelfAttention,
              @SelfAttention,
              @LSHSelfAttention,
              ]
            n_heads = 2
            attn_kv = 64
            dropout = 0.05
            n_tokens = 524288
            # Parameters for SelfAttention:
            # ==============================================================================
            SelfAttention.chunk_len = 64
            SelfAttention.n_chunks_before = 1
            SelfAttention.n_parallel_heads = 1
            SelfAttention.share_qk = False
            # Parameters for LSHSelfAttention:
            # ==============================================================================
            LSHSelfAttention.chunk_len = 64
            LSHSelfAttention.n_buckets = [64, 128]
            LSHSelfAttention.n_chunks_after = 0
            LSHSelfAttention.n_chunks_before = 1
            LSHSelfAttention.n_hashes = 1
            LSHSelfAttention.n_parallel_heads = 1
            LSHSelfAttention.predict_drop_len = 32 # different from original to make code equal
            LSHSelfAttention.predict_mem_len = 64 # different from original to make code equal
            LSHSelfAttention.lsh_seed = 0
            # Parameters for ReformerLM:
            # ==============================================================================
            ReformerLM.attention_type = %attn_type
            ReformerLM.d_attention_key = %attn_kv
            ReformerLM.d_attention_value = %attn_kv
            ReformerLM.d_model = 256
            ReformerLM.d_ff = 512
            ReformerLM.dropout = %dropout
            ReformerLM.ff_activation = @trax.layers.Relu
            ReformerLM.max_len = %n_tokens
            ReformerLM.mode = 'train'
            ReformerLM.n_heads = %n_heads
            ReformerLM.n_layers = %n_layers
            ReformerLM.vocab_size = 320
            ReformerLM.axial_pos_shape = (512, 1024)
            ReformerLM.d_axial_pos_embs= (64, 192)
            """
        )
        trax_model = trax.models.ReformerLM(mode=mode)
        trax_model.init(input_signature)
        trax_model.init_from_file(trax_model_path, weights_only=True)
        return trax_model

    def load_enwik8_model(
        self, trax_model_path, input_signature, mode="predict"
    ):
        gin.parse_config(
            """
            import trax.layers
            import trax.models
            import trax.optimizers
            import trax.supervised.inputs
            import trax.supervised.trainer_lib

            # Macros:
            # ==============================================================================
            attn_kv = 128
            attn_type = \
                [@SelfAttention,
                 @SelfAttention,
                 @LSHSelfAttention,
                 @SelfAttention]
            dropout = 0.2
            n_layers = 12

            # SelfAttention: attend to nearby items
            # Specifying chunk_len restricts an item to attend within its own chunk, as well
            # as the previous `n_chunks_before` chunks.
            SelfAttention.chunk_len = 128
            SelfAttention.n_chunks_before = 1

            # LSHSelfAttention: locality-sensitive hashing (LSH) attention
            LSHSelfAttention.chunk_len = 256
            LSHSelfAttention.n_buckets = 512
            LSHSelfAttention.n_chunks_before = 1
            LSHSelfAttention.n_hashes = 4


            # Parameters for SelfAttention:
            # ==============================================================================
            SelfAttention.attention_dropout = 0.2
            # SelfAttention.chunk_len: see top
            SelfAttention.n_chunks_after = 0
            # SelfAttention.n_chunks_before: see top

            # Parameters for LSHSelfAttention:
            # ==============================================================================
            LSHSelfAttention.attention_dropout = 0.1
            # LSHSelfAttention.chunk_len: see top
            # LSHSelfAttention.n_buckets: see top
            LSHSelfAttention.n_chunks_after = 0
            # LSHSelfAttention.n_chunks_before: see top
            # LSHSelfAttention.n_hashes: see top
            LSHSelfAttention.n_parallel_heads = 1
            LSHSelfAttention.predict_drop_len = 64  # was; 256
            LSHSelfAttention.predict_mem_len = 16384

            # Parameters for ReformerLM:
            # ==============================================================================
            ReformerLM.attention_type = %attn_type
            ReformerLM.d_attention_key = %attn_kv
            ReformerLM.d_attention_value = %attn_kv
            ReformerLM.d_model = 1024
            ReformerLM.d_ff = 4096
            ReformerLM.dropout = %dropout
            ReformerLM.ff_activation = @trax.layers.Relu
            ReformerLM.max_len = 65536
            # ReformerLM.mode = 'train'
            ReformerLM.n_heads = 8
            ReformerLM.n_layers = %n_layers
            ReformerLM.vocab_size = 258  # Includes pad token and unused EOS token
            ReformerLM.axial_pos_shape = (128, 512)
            ReformerLM.d_axial_pos_embs= (256, 768)

            # Parameters for Adam:
            # ==============================================================================
            Adam.b1 = 0.9
            Adam.b2 = 0.98
            Adam.eps = 1e-09
            Adam.weight_decay_rate = 0.0

            # Parameters for batch_fn:
            # ==============================================================================
            batch_fn.batch_size_per_device = 1
            batch_fn.eval_batch_size = 8
            batch_fn.max_eval_length = 65536

            # Parameters for inputs:
            # ==============================================================================
            inputs.data_dir = None
            inputs.dataset_name = 't2t_enwik8_l65k'
            inputs.input_name = 'targets'

            # Parameters for MultifactorSchedule:
            # ==============================================================================
            # 0.03125 ~= 1024^-0.5 = d_model^-0.5
            MultifactorSchedule.constant = 0.03125
            MultifactorSchedule.factors = 'constant * linear_warmup * rsqrt_decay'
            MultifactorSchedule.warmup_steps = 2000

            # Parameters for Adam:
            # ==============================================================================
            Adam.weight_decay_rate=0.0
            Adam.b1 = 0.9
            Adam.b2 = 0.98
            Adam.eps = 1e-9

            # Parameters for train:
            # ==============================================================================
            train.eval_frequency = 500
            train.eval_steps = 8
            """
        )
        trax_model = trax.models.ReformerLM(mode=mode)
        trax_model.init(input_signature)
        trax_model.init_from_file(trax_model_path, weights_only=True)
        return trax_model

    def _set_param(self, torch_layer, weight, bias=None, name=None):
        with torch.no_grad():
            assert (
                torch_layer.weight.shape == weight.shape
            ), "{} layer.weight does not match".format(torch_layer)
            torch_layer.weight = torch.nn.Parameter(weight)
            if bias is not None:
                assert (
                    torch_layer.bias.shape == bias.shape
                ), "{} layer.bias does not match".format(torch_layer)
                torch_layer.bias = torch.nn.Parameter(bias)
        return True

    def _test_param(self, torch_layer, grad, bias_grad=None, name=""):
        assert (
            torch_layer.weight.grad.shape == grad.shape
        ), "{} layer.grad does not match".format(torch_layer)
        if torch.allclose(torch_layer.weight.grad, grad, atol=1e-3):
            print("{}-{} layer.grad is good!".format(name, torch_layer))
        else:
            print("ERROR {}-{} layer.grad is not good!".format(name, torch_layer))
            return False
        if bias_grad is not None:
            assert (
                torch_layer.bias.grad.shape == bias_grad.shape
            ), "{} layer.bias does not match".format(torch_layer)
            if torch.allclose(torch_layer.bias.grad, bias_grad, atol=1e-3):
                print("{}-{} layer.grad bias is good!".format(name, torch_layer))
            else:
                print(
                    "ERROR {}-{} layer.grad bias is not good!".format(name, torch_layer)
                )
                return False
        return True

    def _set_layer_weights_in_torch_lsh(
        self, weights, torch_layer, hidden_size, exec_fn=None
    ):
        all_test_true = True
        if exec_fn is None:
            exec_fn = self._set_param

        # set torch weights for 1-to-1 comparison
        np_query_key = np.asarray(weights[0])
        np_value = np.asarray(weights[1])
        np_dense = np.asarray(weights[2])

        all_test_true = (
            exec_fn(
                torch_layer.self_attention.query_key,
                torch.tensor(np_query_key)
                .transpose(1, 2)
                .contiguous()
                .view(-1, hidden_size),
                name="attn_query_key",
            )
            and all_test_true
        )

        all_test_true = (
            exec_fn(
                torch_layer.self_attention.value,
                torch.tensor(np_value)
                .transpose(1, 2)
                .contiguous()
                .view(-1, hidden_size),
                name="attn_value",
            )
            and all_test_true
        )

        all_test_true = (
            exec_fn(
                torch_layer.output.dense,
                torch.tensor(np_dense)
                .view(-1, hidden_size)
                .contiguous()
                .transpose(0, 1),
                name="attn_dense",
            )
            and all_test_true
        )
        return all_test_true

    def _set_layer_weights_in_torch_local(
        self, weights, torch_layer, hidden_size, exec_fn=None
    ):
        all_test_true = True

        if exec_fn is None:
            exec_fn = self._set_param

        # set torch weights for 1-to-1 comparison
        np_query = np.asarray(weights[0])
        np_key = np.asarray(weights[1])
        np_value = np.asarray(weights[2])
        np_dense = np.asarray(weights[3])

        all_test_true = (
            exec_fn(
                torch_layer.self_attention.query,
                torch.tensor(np_query)
                .transpose(1, 2)
                .contiguous()
                .view(-1, hidden_size),
            )
            and all_test_true
        )
        all_test_true = (
            exec_fn(
                torch_layer.self_attention.key,
                torch.tensor(np_key).transpose(1, 2).contiguous().view(-1, hidden_size),
            )
            and all_test_true
        )
        all_test_true = (
            exec_fn(
                torch_layer.self_attention.value,
                torch.tensor(np_value)
                .transpose(1, 2)
                .contiguous()
                .view(-1, hidden_size),
            )
            and all_test_true
        )
        all_test_true = (
            exec_fn(
                torch_layer.output.dense,
                torch.tensor(np_dense)
                .view(-1, hidden_size)
                .contiguous()
                .transpose(0, 1),
            )
            and all_test_true
        )
        return all_test_true

    def _set_block_weights_in_torch(
        self, weights, torch_block, hidden_size, exec_fn=None
    ):
        all_test_true = True

        if exec_fn is None:
            exec_fn = self._set_param

        # intermediate weighs
        #        intermediate_weights = weights[2][0][2][2]
        intermediate_weights = weights[2][0][1][2]

        # Chunked Feed Forward
        if len(intermediate_weights) == 4:
            intermediate_weights = intermediate_weights[2]

        # intermediate out
        out_dense_weight = np.asarray(intermediate_weights[4][0])
        out_dense_bias = np.asarray(intermediate_weights[4][1])
        all_test_true = (
            exec_fn(
                torch_block.feed_forward.output.dense,
                torch.tensor(out_dense_weight).transpose(0, 1).contiguous(),
                torch.tensor(out_dense_bias),
                name="res_feed_forward_2",
            )
            and all_test_true
        )

        # intermediate dense
        inter_dense_weight = np.asarray(intermediate_weights[1][0])
        inter_dense_bias = np.asarray(intermediate_weights[1][1])
        all_test_true = (
            exec_fn(
                torch_block.feed_forward.dense.dense,
                torch.tensor(inter_dense_weight).transpose(0, 1).contiguous(),
                torch.tensor(inter_dense_bias),
                name="res_feed_forward_1",
            )
            and all_test_true
        )

        # layernorm 2
        layer_norm_2_weight = np.asarray(intermediate_weights[0][0])
        layer_norm_2_bias = np.asarray(intermediate_weights[0][1])
        all_test_true = (
            exec_fn(
                torch_block.feed_forward.layer_norm,
                torch.tensor(layer_norm_2_weight),
                torch.tensor(layer_norm_2_bias),
                name="layer_norm_2",
            )
            and all_test_true
        )

        # lsh weights + output
        attn_weights = weights[0][1]
        if len(attn_weights) < 4:
            all_test_true = (
                self._set_layer_weights_in_torch_lsh(
                    attn_weights, torch_block.attention, hidden_size, exec_fn=exec_fn
                )
                and all_test_true
            )
        else:
            all_test_true = (
                self._set_layer_weights_in_torch_local(
                    attn_weights, torch_block.attention, hidden_size, exec_fn=exec_fn
                )
                and all_test_true
            )

        # layernorm 1
        layer_norm_1 = weights[0][0][0]
        layer_norm_1_weight = np.asarray(layer_norm_1[0])
        layer_norm_1_bias = np.asarray(layer_norm_1[1])
        all_test_true = (
            exec_fn(
                torch_block.attention.layer_norm,
                torch.tensor(layer_norm_1_weight),
                torch.tensor(layer_norm_1_bias),
                name="layer_norm",
            )
            and all_test_true
        )

        return all_test_true

    def _set_model_weights_in_torch(
        self, weights, torch_model, hidden_size, set_params=True
    ):
        # reformer model
        torch_model_reformer = torch_model.reformer

        all_test_true = True
        if set_params is True:
            exec_fn = self._set_param
        else:
            exec_fn = self._test_param

        # output embeddings
        output_embed_weights = np.asarray(weights[9][0])
        output_embed_bias = np.asarray(weights[9][1])
        all_test_true = (
            exec_fn(
                torch_model.lm_head.decoder,
                torch.tensor(output_embed_weights).transpose(0, 1).contiguous(),
                torch.tensor(output_embed_bias),
                name="lm_head",
            )
            and all_test_true
        )

        # output layer norm
        layer_norm_out_weight = np.asarray(weights[7][0])
        layer_norm_out_bias = np.asarray(weights[7][1])
        all_test_true = (
            exec_fn(
                torch_model_reformer.encoder.layer_norm,
                torch.tensor(layer_norm_out_weight),
                torch.tensor(layer_norm_out_bias),
                name="last layer norm",
            )
            and all_test_true
        )

        trax_layer_weights = weights[5]
        assert len(torch_model_reformer.encoder.layers) * 4 == len(
            trax_layer_weights
        ), "HF and trax model do not have the same number of layers"
        for layer_idx, layer in enumerate(torch_model_reformer.encoder.layers[::-1]):
            block_weights = trax_layer_weights[::-1][
                4 * layer_idx : 4 * (layer_idx + 1)
            ][::-1]
            all_test_true = (
                self._set_block_weights_in_torch(
                    block_weights, layer, hidden_size, exec_fn=exec_fn
                )
                and all_test_true
            )

        if isinstance(weights[3], tuple):
            position_embeddings = torch_model_reformer.embeddings.position_embeddings
            for emb_idx in range(len(position_embeddings.weights)):
                emb_weights = np.asarray(weights[3][emb_idx][0])
                assert (
                    position_embeddings.weights[emb_idx].shape == emb_weights.shape
                ), "{} emb does not match".format(position_embeddings[emb_idx])
                if set_params is True:
                    position_embeddings.weights[emb_idx] = torch.nn.Parameter(
                        torch.tensor(emb_weights)
                    )
                else:
                    if torch.allclose(
                        position_embeddings.weights[emb_idx].grad,
                        torch.tensor(emb_weights),
                        atol=1e-3,
                    ):
                        print("{} layer.grad is good!".format(position_embeddings))
                    else:
                        print(
                            "ERROR: {}-{} layer.grad is not good".format(
                                position_embeddings, "axs_pos_embeds"
                            )
                        )

        # word embeds
        word_embeddings = np.asarray(weights[1])
        all_test_true = (
            exec_fn(
                torch_model_reformer.embeddings.word_embeddings,
                torch.tensor(word_embeddings),
                name="word_embed",
            )
            and all_test_true
        )

        return all_test_true
