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
import numpy as np

# trax imports - to be deleted later
import trax
from trax import math as trax_math
from trax.shapes import ShapeDtype as trax_ShapeDtype

import gin

import jax
from trax.layers.research.efficient_attention_v2 import (
    LSHSelfAttention as TraxLSHSelfAttention,
    SelfAttention as TraxSelfAttention,
)
from trax.models.reformer.reformer import DecoderBlock as TraxLSHAttentionBlock
from trax.models.reformer.reformer import ReformerLM as TraxReformer
from trax import layers as tl

from transformers import (
    ReformerAttention,
    ReformerLayer,
    ReformerConfig,
    ReformerModelWithLMHead,
)


from transformers import is_torch_available  # noqa: F401
from .utils import require_torch, torch_device  # noqa: F401


if is_torch_available():
    import torch  # noqa: F401
#    from transformers.modeling_reformer import ()

PATH_TO_SAVE_WEIGHTS = (
    "/home/patrick/hugging_face/experiments/reformer/intermediate_weights"
)


class TraxUtils(object):
    """ class that will help for testing in the beginning
        should be deleted step-by-step

        README (HOW-TO-INSTALL TRAX):
        1) git clone https://github.com/patrickvonplaten/trax.git

           - I had to do one tiny change to make the imports work,
             see: https://github.com/patrickvonplaten/trax/commit/6c23e88afe7f1c57b0c38eeaa4d450e5f912590c)
        2) link your PYTHON_PATH to ~/trax/trax
        3) pip install all the missing packages HINT: the package gin is installed

           - HINT: the package gin is installed with pip install gin-config==0.1.4
                   and not pip install gin.
           - The other packages can just be installed with pip install <package> form
             error message "<package> missing"
    """

    def __init__(self, shape):
        self._shape = shape

    def convert_to_jax_array(self, np_array):
        return jax.numpy.asarray(np_array)

    def get_input_signature(self, shape=None, dtype=trax_math.numpy.float32):
        with trax_math.use_backend("jax"):
            if shape is None:
                shape = self._shape
            input_signature = trax_ShapeDtype(shape, dtype)
        return input_signature

    def get_local_layer(
        self,
        config,
        use_reference_code=True,
        mode="eval",
        path_to_save_weights="/home/patrick/hugging_face/experiments/reformer/intermediate_weights",
        **kwargs
    ):

        with trax_math.use_backend("jax"):
            layer = TraxSelfAttention(
                n_heads=config.num_attention_heads,
                d_qk=config.attention_head_size,
                d_v=config.attention_head_size,
                chunk_len=config.chunk_length,
                n_chunks_before=config.num_chunks_before,
                n_chunks_after=config.num_chunks_after,
                attention_dropout=config.attention_probs_dropout_prob,
                output_dropout=config.hidden_dropout_prob,
                causal=config.is_decoder,
                use_reference_code=use_reference_code,
                mode=mode,
            )
            return layer

    def get_lsh_layer(
        self,
        config,
        use_reference_code=True,
        mode="eval",
        path_to_save_weights="/home/patrick/hugging_face/experiments/reformer/intermediate_weights",
        **kwargs
    ):

        with trax_math.use_backend("jax"):
            layer = TraxLSHSelfAttention(
                n_heads=config.num_attention_heads,
                d_qk=config.attention_head_size,
                d_v=config.attention_head_size,
                chunk_len=config.chunk_length,
                n_chunks_before=config.num_chunks_before,
                n_chunks_after=config.num_chunks_after,
                n_hashes=config.num_hashes,
                n_buckets=config.num_buckets,
                attention_dropout=config.attention_probs_dropout_prob,
                output_dropout=config.hidden_dropout_prob,
                hash_seed=config.seed,
                causal=config.is_decoder,
                use_reference_code=use_reference_code,
                mode=mode,
                path_to_save_weights=path_to_save_weights,
            )
        return layer

    def forward_layer(
        self, np_input_data, layer, input_signature=None, random_number_generator=None,
    ):
        with trax_math.use_backend("jax"):
            input_data = self.convert_to_jax_array(np_input_data)

            if input_signature is None:
                input_signature = self.get_input_signature()

            weights, state = layer.init(input_signature)

            if random_number_generator is None:
                random_number_generator = layer.new_rngs(1)[0]

            output = layer(
                input_data, weights=weights, state=state, rng=random_number_generator
            )

        return output, weights, state

    def get_block(
        self,
        config,
        use_reference_code=True,
        share_qk=True,
        ff_use_sru=0,
        mode="eval",
        path_to_save_weights=PATH_TO_SAVE_WEIGHTS,
    ):

        with trax_math.use_backend("jax"):
            with jax.disable_jit():
                list_of_layers = TraxLSHAttentionBlock(
                    d_model=config.hidden_size,
                    d_ff=config.feed_forward_size,
                    d_attention_key=config.attention_head_size,
                    d_attention_value=config.attention_head_size,
                    n_heads=config.num_attention_heads,
                    n_attention_chunks=None,
                    attention_type=tl.LSHSelfAttention,
                    dropout=config.hidden_dropout_prob,
                    share_qk=share_qk,
                    ff_activation=tl.Gelu,
                    ff_use_sru=ff_use_sru,
                    ff_chunk_size=config.chunk_size_feed_forward,
                    mode=mode,
                    causal=config.is_decoder,
                    chunk_len=config.chunk_length,
                    n_chunks_before=config.num_chunks_before,
                    n_chunks_after=config.num_chunks_after,
                    n_hashes=config.num_hashes,
                    n_buckets=config.num_buckets,
                    use_reference_code=use_reference_code,
                    hash_seed=config.seed,
                    path_to_save_weights=path_to_save_weights,
                )
                block = tl.Serial(tl.ReversibleSerial([list_of_layers]))

        return block

    def forward_block(
        self, np_input_data, block, input_signature=None, random_number_generator=None,
    ):
        with trax_math.use_backend("jax"):
            input_data = self.convert_to_jax_array(np_input_data)
            input_data = (input_data,) * 2

            if input_signature is None:
                input_signature = self.get_input_signature()
                input_signature = (input_signature, input_signature)

            weights, state = block.init(input_signature)

            if random_number_generator is None:
                random_number_generator = block.new_rngs(1)[0]

            output = block(
                input_data, weights=weights, state=state, rng=random_number_generator
            )

        return output, weights, state

    def get_model(
        self,
        config,
        use_reference_code=True,
        share_qk=False,
        ff_use_sru=0,
        mode="eval",
        num_chunks=0,
        n_buckets=None,
        chunk_size_feed_forward=None,
        path_to_save_weights=PATH_TO_SAVE_WEIGHTS,
    ):
        n_buckets = n_buckets if n_buckets is not None else config.num_buckets
        chunk_size_feed_forward = (
            chunk_size_feed_forward
            if chunk_size_feed_forward
            else config.chunk_size_feed_forward
        )
        with trax_math.use_backend("jax"):
            with jax.disable_jit():
                model = TraxReformer(
                    vocab_size=config.vocab_size,
                    d_model=config.hidden_size,
                    d_ff=config.feed_forward_size,
                    d_attention_key=config.attention_head_size,
                    d_attention_value=config.attention_head_size,
                    n_layers=config.num_hidden_layers,
                    n_heads=config.num_attention_heads,
                    dropout=config.hidden_dropout_prob,
                    max_len=config.max_position_embeddings,
                    n_chunks=num_chunks,
                    n_attention_chunks=None,
                    attention_type=tl.LSHSelfAttention,
                    share_qk=share_qk,
                    axial_pos_shape=config.axial_pos_shape,
                    d_axial_pos_embs=config.axial_pos_embds_dim,
                    ff_activation=tl.Gelu,
                    ff_use_sru=ff_use_sru,
                    ff_chunk_size=chunk_size_feed_forward,
                    mode=mode,
                    causal=config.is_decoder,
                    chunk_len=config.chunk_length,
                    n_chunks_before=config.num_chunks_before,
                    n_chunks_after=config.num_chunks_after,
                    n_hashes=config.num_hashes,
                    n_buckets=n_buckets,
                    use_reference_code=use_reference_code,
                    hash_seed=config.seed,
                    path_to_save_weights=path_to_save_weights,
                )

                return model

    def forward_model(
        self,
        np_input_data,
        model,
        input_signature=None,
        random_number_generator=None,
        weights=None,
        state=None,
        only_init=False,
    ):
        with trax_math.use_backend("jax"):
            input_data = self.convert_to_jax_array(np_input_data)
            input_data = (input_data,) * 2

            if input_signature is None:
                input_signature = self.get_input_signature(dtype=trax_math.numpy.int32)
                input_signature = (input_signature, input_signature)

            if weights is None and state is None:
                weights, state = model.init(input_signature)

            if only_init is True:
                return

            if random_number_generator is None:
                random_number_generator = model.new_rngs(1)[0]

            output = model(
                input_data, weights=weights, state=state, rng=random_number_generator
            )

        return output, weights, state


@require_torch
class ReformerIntegrationTests(unittest.TestCase):
    def _set_param(self, torch_layer, weight, bias=None):
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

    def _set_layer_weights_in_torch_lsh(self, weights, torch_layer, hidden_size):
        # set torch weights for 1-to-1 comparison
        np_query_key = np.asarray(weights[0])
        np_value = np.asarray(weights[1])
        np_dense = np.asarray(weights[2])

        self._set_param(
            torch_layer.self_attention.query_key,
            torch.tensor(np_query_key)
            .transpose(1, 2)
            .contiguous()
            .view(-1, hidden_size),
        )
        self._set_param(
            torch_layer.self_attention.value,
            torch.tensor(np_value).transpose(1, 2).contiguous().view(-1, hidden_size),
        )
        self._set_param(
            torch_layer.output.dense,
            torch.tensor(np_dense).view(-1, hidden_size).contiguous().transpose(0, 1),
        )

    def _set_layer_weights_in_torch_local(self, weights, torch_layer, hidden_size):
        # set torch weights for 1-to-1 comparison
        np_query = np.asarray(weights[0])
        np_key = np.asarray(weights[1])
        np_value = np.asarray(weights[2])
        np_dense = np.asarray(weights[3])

        self._set_param(
            torch_layer.self_attention.query,
            torch.tensor(np_query).transpose(1, 2).contiguous().view(-1, hidden_size),
        )
        self._set_param(
            torch_layer.self_attention.key,
            torch.tensor(np_key).transpose(1, 2).contiguous().view(-1, hidden_size),
        )
        self._set_param(
            torch_layer.self_attention.value,
            torch.tensor(np_value).transpose(1, 2).contiguous().view(-1, hidden_size),
        )
        self._set_param(
            torch_layer.output.dense,
            torch.tensor(np_dense).view(-1, hidden_size).contiguous().transpose(0, 1),
        )

    def _set_block_weights_in_torch(self, weights, torch_block, hidden_size):
        # layernorm 1
        layer_norm_1 = weights[0][0][0]
        layer_norm_1_weight = np.asarray(layer_norm_1[0])
        layer_norm_1_bias = np.asarray(layer_norm_1[1])
        self._set_param(
            torch_block.attention.layer_norm,
            torch.tensor(layer_norm_1_weight),
            torch.tensor(layer_norm_1_bias),
        )

        # lsh weights + output
        lsh_weights = weights[0][1]
        self._set_layer_weights_in_torch_lsh(
            lsh_weights, torch_block.attention, hidden_size
        )

        # intermediate weighs
        intermediate_weights = weights[2][0][2][2]

        # Chunked Feed Forward
        if len(intermediate_weights) == 4:
            intermediate_weights = intermediate_weights[2]

        # layernorm 2
        layer_norm_2_weight = np.asarray(intermediate_weights[0][0])
        layer_norm_2_bias = np.asarray(intermediate_weights[0][1])
        self._set_param(
            torch_block.feed_forward.layer_norm,
            torch.tensor(layer_norm_2_weight),
            torch.tensor(layer_norm_2_bias),
        )

        # intermediate dense
        inter_dense_weight = np.asarray(intermediate_weights[1][0])
        inter_dense_bias = np.asarray(intermediate_weights[1][1])
        self._set_param(
            torch_block.feed_forward.dense.dense,
            torch.tensor(inter_dense_weight).transpose(0, 1).contiguous(),
            torch.tensor(inter_dense_bias),
        )

        # intermediate out
        out_dense_weight = np.asarray(intermediate_weights[4][0])
        out_dense_bias = np.asarray(intermediate_weights[4][1])
        self._set_param(
            torch_block.feed_forward.output.dense,
            torch.tensor(out_dense_weight).transpose(0, 1).contiguous(),
            torch.tensor(out_dense_bias),
        )

    def _set_model_weights_in_torch(self, weights, torch_model, hidden_size):
        # reformer model
        torch_model_reformer = torch_model.reformer

        # word embeds
        word_embeddings = np.asarray(weights[1])
        self._set_param(
            torch_model_reformer.embeddings.word_embeddings,
            torch.tensor(word_embeddings),
        )

        if isinstance(weights[3], tuple):
            position_embeddings = torch_model_reformer.embeddings.position_embeddings
            for emb_idx in range(len(position_embeddings.weights)):
                emb_weights = np.asarray(weights[3][emb_idx][0])
                assert (
                    position_embeddings.weights[emb_idx].shape == emb_weights.shape
                ), "{} emb does not match".format(position_embeddings[emb_idx])
                position_embeddings.weights[emb_idx] = torch.nn.Parameter(
                    torch.tensor(emb_weights)
                )

        trax_layer_weights = weights[5]
        assert len(torch_model_reformer.encoder.layer) * 4 + 1 == len(
            trax_layer_weights
        ), "HF and trax model do not have the same number of layers"
        for layer_idx, layer in enumerate(torch_model_reformer.encoder.layer):
            block_weights = trax_layer_weights[4 * layer_idx : 4 * (layer_idx + 1)]
            self._set_block_weights_in_torch(block_weights, layer, hidden_size)

        # output weights
        out_weights = weights[6]

        # output layer norm
        layer_norm_out_weight = np.asarray(out_weights[0][0])
        layer_norm_out_bias = np.asarray(out_weights[0][1])
        self._set_param(
            torch_model_reformer.encoder.layer_norm,
            torch.tensor(layer_norm_out_weight),
            torch.tensor(layer_norm_out_bias),
        )

        # output embeddings
        output_embed_weights = np.asarray(out_weights[2][0])
        output_embed_bias = np.asarray(out_weights[2][1])
        self._set_param(
            torch_model.lm_head.decoder,
            torch.tensor(output_embed_weights).transpose(0, 1).contiguous(),
            torch.tensor(output_embed_bias),
        )

        pass

    def test_lsh_layer(self):
        config = ReformerConfig()
        shape = (2, 14, config.hidden_size)  # Batch x SeqLen x hiddenSize
        np_input = np.random.rand(*shape)

        trax_utils = TraxUtils(shape)
        trax_layer = trax_utils.get_lsh_layer(config)
        trax_output, trax_weights, trax_state = trax_utils.forward_layer(
            np_input, layer=trax_layer
        )

        hf_input = torch.tensor(np_input, dtype=torch.float)
        hf_layer = ReformerAttention(config)
        self._set_layer_weights_in_torch_lsh(trax_weights, hf_layer, config.hidden_size)
        hf_layer.eval()

        hf_attention_all_heads = hf_layer.self_attention(hf_input)[0]
        hf_output = hf_layer.output(hf_attention_all_heads, torch.zeros_like(hf_input))

        trax_torch_output = torch.tensor(np.asarray(trax_output))
        self.assertTrue(torch.allclose(hf_output, trax_torch_output, atol=1e-3))

    def test_local_layer(self):
        config = ReformerConfig()
        shape = (2, 14, config.hidden_size)  # Batch x SeqLen x hiddenSize
        np_input = np.random.rand(*shape)

        trax_utils = TraxUtils(shape)
        trax_layer = trax_utils.get_local_layer(config)
        trax_output, trax_weights, trax_state = trax_utils.forward_layer(
            np_input, layer=trax_layer
        )

        hf_input = torch.tensor(np_input, dtype=torch.float)
        config.attn_type = "local"
        hf_layer = ReformerAttention(config)
        self._set_layer_weights_in_torch_local(
            trax_weights, hf_layer, config.hidden_size
        )
        hf_layer.eval()

        hf_attention_all_heads = hf_layer.self_attention(hf_input)[0]
        hf_output = hf_layer.output(hf_attention_all_heads, torch.zeros_like(hf_input))

        trax_torch_output = torch.tensor(np.asarray(trax_output))
        self.assertTrue(torch.allclose(hf_output, trax_torch_output, atol=1e-3))

    def test_lsh_block(self):
        config = ReformerConfig()

        shape = (2, 7, config.hidden_size)  # Batch x SeqLen x ModelDimPerHead
        np_input = np.random.rand(*shape)

        trax_utils = TraxUtils(shape)
        trax_block = trax_utils.get_block(config)
        trax_output, trax_weights, trax_state = trax_utils.forward_block(
            np_input, block=trax_block
        )
        trax_torch_output_1 = torch.tensor(np.asarray(trax_output[0]))
        trax_torch_output_2 = torch.tensor(np.asarray(trax_output[1]))

        hf_input = torch.tensor(np_input, dtype=torch.float)
        hf_block = ReformerLayer(config)
        self._set_block_weights_in_torch(trax_weights[0], hf_block, config.hidden_size)
        hf_block.eval()

        hf_output_1, hf_output_2 = hf_block(hf_input, hf_input)[:2]

        self.assertTrue(torch.allclose(hf_output_1, trax_torch_output_1, atol=1e-3))
        self.assertTrue(torch.allclose(hf_output_2, trax_torch_output_2, atol=1e-3))

    def test_reformer_lm_model(self):
        config = ReformerConfig()

        shape = (2, 14)  # Batch x SeqLen x ModelDimPerHead
        np_input = np.random.randint(0, config.vocab_size, size=shape)
        np_zeros = np.zeros((shape[0], 1), dtype=np.int)

        trax_utils = TraxUtils(shape)
        trax_model = trax_utils.get_model(config)
        trax_output, trax_weights, trax_state = trax_utils.forward_model(
            np_input, model=trax_model
        )
        trax_torch_output = torch.tensor(np.asarray(trax_output[0]))

        hf_input = torch.cat(
            [torch.tensor(np_zeros), torch.tensor(np_input[:, :-1])], dim=-1
        )
        hf_model = ReformerModelWithLMHead(config)
        self._set_model_weights_in_torch(trax_weights, hf_model, config.hidden_size)
        hf_model.eval()

        hf_output = hf_model(hf_input)
        log_softmax_output = torch.nn.functional.log_softmax(hf_output[0], dim=-1)

        self.assertTrue(
            torch.allclose(log_softmax_output, trax_torch_output, atol=1e-3)
        )

    def test_pretrained_crime_and_punishment_lm_model(self):
        hf_model = ReformerModelWithLMHead.from_pretrained(
            "patrickvonplaten/reformer-crime-and-punish"
        )
        config = hf_model.config

        trax_model_path = (
            "/home/patrick/hugging_face/models/trained_reformer_colab/model.pkl"
        )

        trax_model = self.load_model(trax_model_path)

        shape = (3, 1024)
        np_input = np.random.randint(0, config.vocab_size, size=shape)
        np_zeros = np.zeros((shape[0], 1), dtype=np.int)

        trax_utils = TraxUtils(shape)
        trax_model = trax_utils.get_model(
            config, mode="predict", n_buckets=[64, 128], chunk_size_feed_forward=0
        )

        weights, state = trax_model.init(
            trax_utils.get_input_signature(dtype=trax_math.numpy.int32)
        )

        trax_output = trax_model(np_input)

        trax_torch_output = torch.tensor(np.asarray(trax_output[0]))

        hf_input = torch.cat(
            [torch.tensor(np_zeros), torch.tensor(np_input[:, :-1])], dim=-1
        )
        hf_output = hf_model(hf_input)
        log_softmax_output = torch.nn.functional.log_softmax(hf_output[0], dim=-1)

        self.assertTrue(
            torch.allclose(log_softmax_output, trax_torch_output, atol=1e-3)
        )

    def load_model(self, trax_model_path):
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
            share_qk = False  # LSH attention ignores this flag and always shares q & k
            n_heads = 2
            attn_kv = 64
            dropout = 0.05
            n_tokens = 524288

            # Parameters for MultifactorSchedule:
            # ==============================================================================
            MultifactorSchedule.constant = 0.01
            MultifactorSchedule.factors = 'constant * linear_warmup * cosine_decay'
            MultifactorSchedule.warmup_steps = 100
            MultifactorSchedule.steps_per_cycle = 900

            # Parameters for Adam:
            # ==============================================================================
            Adam.weight_decay_rate=0.0
            Adam.b1 = 0.86
            Adam.b2 = 0.92
            Adam.eps = 1e-9

            # Parameters for SelfAttention:
            # ==============================================================================
            SelfAttention.attention_dropout = 0.05
            SelfAttention.chunk_len = 64
            SelfAttention.n_chunks_before = 1
            SelfAttention.n_parallel_heads = 1

            # Parameters for LSHSelfAttention:
            # ==============================================================================
            LSHSelfAttention.attention_dropout = 0.0
            LSHSelfAttention.chunk_len = 64
            LSHSelfAttention.n_buckets = [64, 128]
            LSHSelfAttention.n_chunks_after = 0
            LSHSelfAttention.n_chunks_before = 1
            LSHSelfAttention.n_hashes = 1
            LSHSelfAttention.n_parallel_heads = 1
            LSHSelfAttention.predict_drop_len = 128
            LSHSelfAttention.predict_mem_len = 1024

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
            ReformerLM.share_qk = %share_qk
            ReformerLM.axial_pos_shape = (512, 1024)
            ReformerLM.d_axial_pos_embs= (64, 192)
            """
        )
        trax_model = trax.models.ReformerLM(mode="predict")
        import ipdb
        ipdb.set_trace()
        trax_model.init_from_file(trax_model_path)
        return trax_model
