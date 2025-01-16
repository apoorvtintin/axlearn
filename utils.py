import numbers
import os
import shutil
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, Callable, NamedTuple, Optional, TypeVar, Union

import jax
import numpy as np
import torch
from jax import numpy as jnp
from transformers import AutoConfig, LlamaForCausalLM

from axlearn.common import config, evaler, input_tf_data, measurement, utils
from axlearn.common.checkpointer import Checkpointer, CheckpointValidationType
from axlearn.common.decoder import LmHead
from axlearn.common.inference import InferenceRunner, _InferenceRunnerState
from axlearn.common.trainer import select_mesh_config
from axlearn.common.utils import PartitionSpec, TensorSpec, MeshShape, infer_mesh_shape
from axlearn.experiments import get_named_trainer_config
from axlearn.experiments.text.gpt import c4_trainer

seed = 123
backend = jax.default_backend()


def get_trainer_config(config_name):
    trainer_config_fn = get_named_trainer_config(
        config_name, config_module="axlearn.experiments.text.gpt.c4_trainer"
    )
    trainer_config = trainer_config_fn()

    # Enable forward pass to return logits and loss
    trainer_config.evalers["validation"].metric_calculator = trainer_config.evalers[
        "validation"
    ].metric_calculator.set(model_method_kwargs={"return_aux": True})

    if backend == "neuron":
        mesh_selector = "neuron-trn2.48xlarge-64"
    else:
        mesh_selector = None

    # extra config from launch_trainer.py get_trainer_config
    if mesh_selector is not None:
        select_mesh_config(trainer_config, mesh_selector=mesh_selector)
    trainer_config.mesh_axis_names = trainer_config.mesh_axis_names or ("data", "model")
    trainer_config.mesh_shape = trainer_config.mesh_shape or (len(jax.devices()), 1)
    if isinstance(trainer_config.mesh_shape, MeshShape):
        trainer_config.mesh_shape = infer_mesh_shape(trainer_config.mesh_shape)

    os.environ["DATA_DIR"] = "gs://axlearn-public/tensorflow_datasets"
    return trainer_config


def get_mesh(trainer_config):
    devices = utils.create_device_mesh(mesh_shape=trainer_config.mesh_shape)
    mesh = jax.sharding.Mesh(devices, trainer_config.mesh_axis_names)
    return mesh


def init_infer_runner(trainer_config, checkpoint_path):
    # trainer_config.set(dir=CHECKPOINT_PATH)

    devices = utils.create_device_mesh(mesh_shape=trainer_config.mesh_shape)
    infer_runner_config = InferenceRunner.config_from_trainer(trainer_config)
    infer_runner_config.init_state_builder.set(dir=checkpoint_path)
    infer_runner = infer_runner_config.instantiate(parent=None)
    return infer_runner, infer_runner_config


def get_checkpointer(checkpoint_path):
    checkpointer_config = Checkpointer.default_config()
    checkpointer_config.dir = checkpoint_path or CHECKPOINT_PATH
    checkpointer_config.name = "checkpointer"
    checkpointer = checkpointer_config.instantiate(parent=None)
    return checkpointer


def load_checkpoint(trainer_config, checkpoint_path, step=None):
    """
    Note: it is ok to use checkpointer to load checkpoint directly
    but need to be careful abou the checkpoint file structure
    if the checkpoint is created by trainer, then it will be structured as following
    step_00034000
      - gda
          - learner
          - model
          - prng_key
    if the checkpoint is create by running checkpointer.save on the model state, it will nto have learner and prng_key
    step_00022794
      - gda
          - decoder
    to keep them consitent, wrap model state in a _InferenceRunnerState

    To load a model only checkpoint folder, can do the following

    checkpointer = get_checkpointer(checkpoint_path)
    prng_key = jax.random.PRNGKey(seed)
    state = model.initialize_parameters_recursively(prng_key=prng_key)
    step, state = checkpointer.restore(state=state, step=step)
    """
    inference_runner, _ = init_infer_runner(trainer_config, checkpoint_path)
    model_state = inference_runner._inference_runner_state.model
    return model_state


def save_axlearn_checkpoint(model, state, checkpoint_path, mesh):
    with mesh:
        checkpointer = get_checkpointer(checkpoint_path)
        inference_runner_state = _InferenceRunnerState(
            # prng_key=TensorSpec(dtype=jnp.uint32, shape=[4], mesh_axes=PartitionSpec(None)),
            # the actual value does not matter since this is just to save the checkpoint in trainer format
            prng_key=jnp.asarray([744862133, 117316191, 744862143, 1173516191], dtype=jnp.uint32),
            model=state,
        )._asdict()
        checkpointer.save(step=22794, state=inference_runner_state)


def save_transformers_checkpoint(model, checkpoint_path):
    model.save_pretrained(checkpoint_path)


converted_tokenizer_path = "fuji_tokenizer"


def convert_tokenizer(
    sentencepiece_model_path="axlearn/data/tokenizers/sentencepiece/bpe_32k_c4.model",
):
    if os.path.isdir(converted_tokenizer_path):
        return

    tokenizer = LlamaTokenizer.from_pretrained(
        sentencepiece_tokenizer_path,
        bos_token="</s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        use_fast=False,
        add_bos_token=True,
    )
    tokenizer.save_pretrained(converted_tokenizer_path)


def copy_files(llama_model_name, checkpoint_path):
    for file_name in os.listdir(converted_tokenizer_path):
        shutil.copyfile(
            os.path.join(converted_tokenizer_path, file_name),
            os.path.join(checkpoint_path, file_name),
        )

    shutil.copyfile(
        os.path.join(llama_model_name, "generation_config.json"),
        os.path.join(checkpoint_path, "generation_config.json"),
    )


def jax_to_torch(jax_tensor):
    """Convert Jax tensor to Torch Tensor."""
    array = np.array(jax_tensor)
    return torch.from_numpy(array)


def as_jax_tensor(x: Any):
    """Converts `x` to jax.Array recursively.

    Args:
        x: a jnp array, numpy array, TF/PyTorch Tensor, or a nested structure of arrays or Tensors.

    Returns:
        A nested structure with the same structure as `x` but with values converted to jax.Arrays.

    Raises:
        NotImplementedError: If conversion for the input type is unsupported.
    """
    if isinstance(x, jax.Array):
        return x
    if isinstance(x, (numbers.Number, np.ndarray)):
        return jnp.asarray(x)
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "numpy"):
        return jnp.asarray(x.numpy())
    if isinstance(x, (Mapping, Sequence)):
        return jax.tree.map(as_jax_tensor, x)
    raise NotImplementedError(f"{type(x)}: {x}")


def as_torch_tensor(x: Any):
    """Converts `x` to torch Tensor recursively.

    Args:
        x: a jnp array, numpy array, TF/PyTorch Tensor, or a nested structure of arrays or Tensors.

    Returns:
        A nested structure with the same structure as `x` but with values converted to torch Tensors.

    Raises:
        NotImplementedError: If conversion for the input type is unsupported.
    """
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, jax.Array):
        return jax_to_torch(x)
    if isinstance(x, (numbers.Number, np.ndarray)):
        return torch.Tensor(x)
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "numpy"):
        return torch.from_numpy(x.numpy())
    if isinstance(x, (Mapping, Sequence)):
        return jax.tree.map(as_torch_tensor, x)
    raise NotImplementedError(f"{type(x)}: {x}")


def transformers_rope_to_axlearn_rope(vector: torch.Tensor) -> torch.Tensor:
    """Permutes q and k vector because transformers package has a different implementation of RoPE.

    The revert operation of the following:
    https://github.com/huggingface/transformers/blob/e42587f596181396e1c4b63660abf0c736b10dae/src/transformers/models/llama/convert_llama_weights_to_hf.py#L136
    """
    n, h, d = vector.shape
    vector = vector.view(n, 2, h // 2, d).transpose(1, 2)
    return vector.reshape(n, h, d)


def axlearn_rope_to_transformers_rope(vector: jax.Array) -> jax.Array:
    """Reverse of transformers_rope_to_axlearn_rope function."""
    n, h, d = vector.shape
    vector = vector.view(n, h // 2, 2, d).transpose(1, 2)
    return vector.reshape(n, h, d)


def parameters_from_llama(
    llama: LlamaForCausalLM, state: dict, use_gqa=False, trn_checkpoint=False
) -> dict:
    """Converts llama weights from huggingface model to fuji state.
    Conversion for llama2 and llama3 would be different. For example 7B model does not use GQA,
    and fuji 7B model also share weights between lm_head and emb.

    The following model are supported and tested:
    - (fuji_model_name="fuji-7B-v2", llama_model_name="Llama-2-7b-hf")
    - (fuji_model_name="fuji-70B-v2", llama_model_name="Llama-2-70b-hf")

    Args:
        llama: A Llama model with type LlamaForCausalLM.
        state: The state of a fuji model.

    Returns:
        NestedTensor containing the same structure as state, but the weights are from llama.
    """
    if trn_checkpoint:
        return _parameters_from_llama_trn(llama, state, use_gqa)
    else:
        return _parameters_from_llama(llama, state, use_gqa)


def _parameters_from_llama(llama: LlamaForCausalLM, state: dict, use_gqa=False) -> dict:
    """Convert llama model weight to fuji model weight.

    Conversion for checkpoint trained in TRN and GPU are different since TRN replaced
    RepeatedTransformerLayer with StackedTransformerLayer and FusedQKVLinear with GroupedQKVLinear.
    """
    # Copy the nested dict. No need to deep copy the data since it will be replaced.
    state = jax.tree.map(lambda x: x, state)
    if "lm_head" in state["decoder"]:
        if id(llama.model.embed_tokens.weight) == id(llama.lm_head.weight):
            raise ValueError("The embed_tokens and lm_head should not share weights.")
        state["decoder"]["lm_head"]["weight"] = llama.lm_head.weight
    elif id(llama.model.embed_tokens.weight) != id(llama.lm_head.weight):
        raise ValueError("The embed_tokens and lm_head should share weights")

    state["decoder"]["emb"]["token_emb"]["weight"] = llama.model.embed_tokens.weight
    gate_proj = []
    up_proj = []
    down_proj = []
    qkv = []
    o = []
    input_norm = []
    post_attention_norm = []
    o_shape = state["decoder"]["transformer"]["repeat"]["layer"]["self_attention"]["attention"][
        "o_proj"
    ]["weight"].shape
    i_shape = state["decoder"]["transformer"]["repeat"]["layer"]["self_attention"]["attention"][
        "i_proj"
    ]["i_proj"]["qkv_proj"][
        "weight"
    ].shape  # (n_layers, hidden_dimension, num_head, head_dimension)

    for layer in llama.model.layers:
        gate_proj.append(layer.mlp.gate_proj.weight.transpose(0, 1))
        up_proj.append(layer.mlp.up_proj.weight.transpose(0, 1))
        down_proj.append(layer.mlp.down_proj.weight.transpose(0, 1))

        # Llama3 and Llama2 70B uses GQA, but Llama2 7B does not
        if use_gqa:
            vector = torch.concat(
                [
                    transformers_rope_to_axlearn_rope(
                        layer.self_attn.q_proj.weight.reshape(-1, i_shape[-1], i_shape[-3])
                    ),
                    transformers_rope_to_axlearn_rope(
                        layer.self_attn.k_proj.weight.reshape(-1, i_shape[-1], i_shape[-3])
                    ),
                    layer.self_attn.v_proj.weight.reshape(-1, i_shape[-1], i_shape[-3]),
                ],
                dim=0,
            ).permute(2, 0, 1)
        else:
            vector = torch.stack(
                [
                    transformers_rope_to_axlearn_rope(
                        layer.self_attn.q_proj.weight.reshape(-1, i_shape[-1], i_shape[-3])
                    ),
                    transformers_rope_to_axlearn_rope(
                        layer.self_attn.k_proj.weight.reshape(-1, i_shape[-1], i_shape[-3])
                    ),
                    layer.self_attn.v_proj.weight.reshape(-1, i_shape[-1], i_shape[-3]),
                ],
                dim=0,
            ).permute([0, 3, 1, 2])
        qkv.append(vector)
        o.append(layer.self_attn.o_proj.weight.reshape(-1, o_shape[-2], o_shape[-1]))
        input_norm.append(layer.input_layernorm.weight)
        post_attention_norm.append(layer.post_attention_layernorm.weight)
    state["decoder"]["transformer"]["repeat"]["layer"]["feed_forward"]["linear1_0"]["weight"] = (
        torch.stack(gate_proj)
    )
    state["decoder"]["transformer"]["repeat"]["layer"]["feed_forward"]["linear1_1"]["weight"] = (
        torch.stack(up_proj)
    )
    state["decoder"]["transformer"]["repeat"]["layer"]["feed_forward"]["linear2"]["weight"] = (
        torch.stack(down_proj)
    )
    state["decoder"]["transformer"]["repeat"]["layer"]["self_attention"]["attention"]["o_proj"][
        "weight"
    ] = torch.stack(o)
    state["decoder"]["transformer"]["repeat"]["layer"]["self_attention"]["attention"]["i_proj"][
        "i_proj"
    ]["qkv_proj"]["weight"] = torch.stack(qkv)
    state["decoder"]["transformer"]["repeat"]["layer"]["self_attention"]["norm"]["scale"] = (
        torch.stack(input_norm)
    )
    state["decoder"]["transformer"]["repeat"]["layer"]["feed_forward"]["norm"]["scale"] = (
        torch.stack(post_attention_norm)
    )
    state["decoder"]["output_norm"]["scale"] = llama.model.norm.weight
    return as_jax_tensor(state)


def _parameters_from_llama_trn(llama: LlamaForCausalLM, state: dict, use_gqa=False) -> dict:
    # Copy the nested dict. No need to deep copy the data since it will be replaced.
    state = jax.tree.map(lambda x: x, state)
    if "lm_head" in state["decoder"]:
        if id(llama.model.embed_tokens.weight) == id(llama.lm_head.weight):
            raise ValueError("The embed_tokens and lm_head should not share weights.")
        state["decoder"]["lm_head"]["weight"] = llama.lm_head.weight
    elif id(llama.model.embed_tokens.weight) != id(llama.lm_head.weight):
        raise ValueError("The embed_tokens and lm_head should share weights")

    state["decoder"]["emb"]["token_emb"]["weight"] = llama.model.embed_tokens.weight
    o_shape = state["decoder"]["transformer"]["layer0"]["self_attention"]["attention"]["o_proj"][
        "weight"
    ].shape

    for idx, layer in enumerate(llama.model.layers):
        state["decoder"]["transformer"][f"layer{idx}"]["feed_forward"]["linear1_0"]["weight"] = (
            layer.mlp.gate_proj.weight.transpose(0, 1)
        )
        state["decoder"]["transformer"][f"layer{idx}"]["feed_forward"]["linear1_1"]["weight"] = (
            layer.mlp.up_proj.weight.transpose(0, 1)
        )
        state["decoder"]["transformer"][f"layer{idx}"]["feed_forward"]["linear2"]["weight"] = (
            layer.mlp.down_proj.weight.transpose(0, 1)
        )

        # Llama3 and Llama2 70B uses GQA, but Llama2 7B does not
        # TODO a more proper naming would be replacing use_gqa with fuse_qkv for this function
        if use_gqa:
            # Do NOT use i_shape[-2] in the following for qkv since they should be different when using GQA
            i_shape = state["decoder"]["transformer"]["layer0"]["self_attention"]["attention"][
                "i_proj"
            ]["i_proj"]["q_proj"]["weight"].shape
            state["decoder"]["transformer"][f"layer{idx}"]["self_attention"]["attention"]["i_proj"][
                "i_proj"
            ]["q_proj"]["weight"] = transformers_rope_to_axlearn_rope(
                layer.self_attn.q_proj.weight.reshape(-1, i_shape[-1], i_shape[-3])
            ).permute(
                2, 0, 1
            )
            state["decoder"]["transformer"][f"layer{idx}"]["self_attention"]["attention"]["i_proj"][
                "i_proj"
            ]["k_proj"]["weight"] = transformers_rope_to_axlearn_rope(
                layer.self_attn.k_proj.weight.reshape(-1, i_shape[-1], i_shape[-3])
            ).permute(
                2, 0, 1
            )
            state["decoder"]["transformer"][f"layer{idx}"]["self_attention"]["attention"]["i_proj"][
                "i_proj"
            ]["v_proj"]["weight"] = layer.self_attn.v_proj.weight.reshape(
                -1, i_shape[-1], i_shape[-3]
            ).permute(
                2, 0, 1
            )
        else:
            i_shape = state["decoder"]["transformer"]["layer0"]["self_attention"]["attention"][
                "i_proj"
            ]["i_proj"]["qkv_proj"]["weight"].shape
            state["decoder"]["transformer"][f"layer{idx}"]["self_attention"]["attention"]["i_proj"][
                "i_proj"
            ]["qkv_proj"]["weight"] = torch.stack(
                [
                    transformers_rope_to_axlearn_rope(
                        layer.self_attn.q_proj.weight.reshape(-1, i_shape[-1], i_shape[-3])
                    ),
                    transformers_rope_to_axlearn_rope(
                        layer.self_attn.k_proj.weight.reshape(-1, i_shape[-1], i_shape[-3])
                    ),
                    layer.self_attn.v_proj.weight.reshape(-1, i_shape[-1], i_shape[-3]),
                ],
                dim=0,
            ).permute(
                [0, 3, 1, 2]
            )
        state["decoder"]["transformer"][f"layer{idx}"]["self_attention"]["attention"]["o_proj"][
            "weight"
        ] = layer.self_attn.o_proj.weight.reshape(-1, o_shape[-2], o_shape[-1])
        state["decoder"]["transformer"][f"layer{idx}"]["self_attention"]["norm"][
            "scale"
        ] = layer.input_layernorm.weight
        state["decoder"]["transformer"][f"layer{idx}"]["feed_forward"]["norm"][
            "scale"
        ] = layer.post_attention_layernorm.weight

    state["decoder"]["output_norm"]["scale"] = llama.model.norm.weight
    return as_jax_tensor(state)


def parameters_to_llama(state: dict, llama, use_gqa=False, trn_checkpoint=False) -> dict:
    """Convert parameters from axlearn fuji to transformers llama.

    Note: jax_to_torch on each jax tensor to make sure the tensors are offloaded to CPU,
    otherwise will run into OOM issue when the model is large
    """
    if trn_checkpoint:
        return _parameters_to_llama_trn(state, llama, use_gqa)
    else:
        return _parameters_to_llama(state, llama, use_gqa)


def _parameters_to_llama(state: dict, llama, use_gqa=False) -> dict:
    """Convert parameters from axlearn fuji to transformers llama."""
    state = jax.tree.map(lambda x: x, state)
    llama_state = llama.state_dict()
    num_layers = llama.config.num_hidden_layers
    hidden_size = llama.config.hidden_size
    num_kv_heads = llama.config.num_key_value_heads

    if "lm_head" in state["decoder"]:
        llama_state["lm_head.weight"] = jax_to_torch(state["decoder"]["lm_head"]["weight"])
    else:
        # if fuji has no lm_head, should share the weights with embed
        llama_state["lm_head.weight"] = jax_to_torch(state["decoder"]["emb"]["token_emb"]["weight"])

    llama_state["model.embed_tokens.weight"] = jax_to_torch(
        state["decoder"]["emb"]["token_emb"]["weight"]
    )

    for idx in range(num_layers):
        # convert linear layers
        llama_state[f"model.layers.{idx}.mlp.gate_proj.weight"] = jax_to_torch(
            state["decoder"]["transformer"]["repeat"]["layer"]["feed_forward"]["linear1_0"][
                "weight"
            ][idx].transpose()
        )
        llama_state[f"model.layers.{idx}.mlp.up_proj.weight"] = jax_to_torch(
            state["decoder"]["transformer"]["repeat"]["layer"]["feed_forward"]["linear1_1"][
                "weight"
            ][idx].transpose()
        )
        llama_state[f"model.layers.{idx}.mlp.down_proj.weight"] = jax_to_torch(
            state["decoder"]["transformer"]["repeat"]["layer"]["feed_forward"]["linear2"]["weight"][
                idx
            ].transpose()
        )

        # convert attention layers
        if use_gqa:
            qkv = jax_to_torch(
                jnp.permute_dims(
                    state["decoder"]["transformer"]["repeat"]["layer"]["self_attention"][
                        "attention"
                    ]["i_proj"]["i_proj"]["qkv_proj"]["weight"][idx],
                    (1, 2, 0),
                )
            )
            # concat order q, k, v
            llama_state[f"model.layers.{idx}.self_attn.q_proj.weight"] = (
                axlearn_rope_to_transformers_rope(qkv[: -2 * num_kv_heads]).reshape(-1, hidden_size)
            )
            llama_state[f"model.layers.{idx}.self_attn.k_proj.weight"] = (
                axlearn_rope_to_transformers_rope(qkv[-2 * num_kv_heads : -num_kv_heads]).reshape(
                    -1, hidden_size
                )
            )
            llama_state[f"model.layers.{idx}.self_attn.v_proj.weight"] = qkv[
                -num_kv_heads:
            ].reshape(-1, hidden_size)
        else:
            qkv = jax_to_torch(
                jnp.permute_dims(
                    state["decoder"]["transformer"]["repeat"]["layer"]["self_attention"][
                        "attention"
                    ]["i_proj"]["i_proj"]["qkv_proj"]["weight"][idx],
                    (0, 2, 3, 1),
                )
            )
            # assert jnp.array_equal(as_jax_tensor(llama_state[f"model.layers.{idx}.self_attn.q_proj.weight"]), axlearn_rope_to_transformers_rope(torch.from_numpy(np.array(qkv[0]))).reshape(-1, hidden_size))
            llama_state[f"model.layers.{idx}.self_attn.q_proj.weight"] = (
                axlearn_rope_to_transformers_rope(qkv[0]).reshape(-1, hidden_size)
            )
            llama_state[f"model.layers.{idx}.self_attn.k_proj.weight"] = (
                axlearn_rope_to_transformers_rope(qkv[1]).reshape(-1, hidden_size)
            )
            llama_state[f"model.layers.{idx}.self_attn.v_proj.weight"] = qkv[2].reshape(
                -1, hidden_size
            )

        # convert attention layer projection
        llama_state[f"model.layers.{idx}.self_attn.o_proj.weight"] = jax_to_torch(
            state["decoder"]["transformer"]["repeat"]["layer"]["self_attention"]["attention"][
                "o_proj"
            ]["weight"][idx].reshape(hidden_size, -1)
        )

        # convert normalization layers
        llama_state[f"model.layers.{idx}.input_layernorm.weight"] = jax_to_torch(
            state["decoder"]["transformer"]["repeat"]["layer"]["self_attention"]["norm"]["scale"][
                idx
            ]
        )
        llama_state[f"model.layers.{idx}.post_attention_layernorm.weight"] = jax_to_torch(
            state["decoder"]["transformer"]["repeat"]["layer"]["feed_forward"]["norm"]["scale"][idx]
        )

    # convert normalization layer
    llama_state["model.norm.weight"] = jax_to_torch(state["decoder"]["output_norm"]["scale"])

    return as_torch_tensor(llama_state)


def _parameters_to_llama_trn(state: dict, llama, use_gqa=False) -> dict:
    """TRN has not been able to train the fuji model with the same config as GPU. 2 changes were made:
    1. replace RepeatTransformerLayer with StackedTransformerLayer
    2. replace FuseGroupedQKVLinear with GroupedQKVLinear
    """
    state = jax.tree.map(lambda x: x, state)
    llama_state = llama.state_dict()
    num_layers = llama.config.num_hidden_layers
    hidden_size = llama.config.hidden_size
    num_kv_heads = llama.config.num_key_value_heads

    if "lm_head" in state["decoder"]:
        llama_state["lm_head.weight"] = jax_to_torch(state["decoder"]["lm_head"]["weight"])
    else:
        # if fuji has no lm_head, should share the weights with embed
        llama_state["lm_head.weight"] = jax_to_torch(state["decoder"]["emb"]["token_emb"]["weight"])

    llama_state["model.embed_tokens.weight"] = jax_to_torch(
        state["decoder"]["emb"]["token_emb"]["weight"]
    )

    for idx in range(num_layers):
        # convert linear layers
        llama_state[f"model.layers.{idx}.mlp.gate_proj.weight"] = jax_to_torch(
            state["decoder"]["transformer"][f"layer{idx}"]["feed_forward"]["linear1_0"][
                "weight"
            ].transpose()
        )
        llama_state[f"model.layers.{idx}.mlp.up_proj.weight"] = jax_to_torch(
            state["decoder"]["transformer"][f"layer{idx}"]["feed_forward"]["linear1_1"][
                "weight"
            ].transpose()
        )
        llama_state[f"model.layers.{idx}.mlp.down_proj.weight"] = jax_to_torch(
            state["decoder"]["transformer"][f"layer{idx}"]["feed_forward"]["linear2"][
                "weight"
            ].transpose()
        )

        # convert attention layers
        # in Apoorv's branch, 70B model uses GroupedQKVLinear and 7B model uses FusedQKVLinear
        # TODO a more proper naming would be replacing use_gqa with fuse_qkv for this function
        # fuse_qkv = not use_gqa
        if use_gqa:
            q = jax_to_torch(
                jnp.permute_dims(
                    state["decoder"]["transformer"][f"layer{idx}"]["self_attention"]["attention"][
                        "i_proj"
                    ]["i_proj"]["q_proj"]["weight"],
                    (1, 2, 0),
                )
            )
            k = jax_to_torch(
                jnp.permute_dims(
                    state["decoder"]["transformer"][f"layer{idx}"]["self_attention"]["attention"][
                        "i_proj"
                    ]["i_proj"]["k_proj"]["weight"],
                    (1, 2, 0),
                )
            )
            v = jax_to_torch(
                jnp.permute_dims(
                    state["decoder"]["transformer"][f"layer{idx}"]["self_attention"]["attention"][
                        "i_proj"
                    ]["i_proj"]["v_proj"]["weight"],
                    (1, 2, 0),
                )
            )
            llama_state[f"model.layers.{idx}.self_attn.q_proj.weight"] = (
                axlearn_rope_to_transformers_rope(q).reshape(-1, hidden_size)
            )
            llama_state[f"model.layers.{idx}.self_attn.k_proj.weight"] = (
                axlearn_rope_to_transformers_rope(k).reshape(-1, hidden_size)
            )
            llama_state[f"model.layers.{idx}.self_attn.v_proj.weight"] = v.reshape(-1, hidden_size)
        else:
            qkv = jax_to_torch(
                jnp.permute_dims(
                    state["decoder"]["transformer"][f"layer{idx}"]["self_attention"]["attention"][
                        "i_proj"
                    ]["i_proj"]["qkv_proj"]["weight"],
                    (0, 2, 3, 1),
                )
            )
            llama_state[f"model.layers.{idx}.self_attn.q_proj.weight"] = (
                axlearn_rope_to_transformers_rope(qkv[0]).reshape(-1, hidden_size)
            )
            llama_state[f"model.layers.{idx}.self_attn.k_proj.weight"] = (
                axlearn_rope_to_transformers_rope(qkv[1]).reshape(-1, hidden_size)
            )
            llama_state[f"model.layers.{idx}.self_attn.v_proj.weight"] = qkv[2].reshape(
                -1, hidden_size
            )

        # convert attention layer projection
        llama_state[f"model.layers.{idx}.self_attn.o_proj.weight"] = jax_to_torch(
            state["decoder"]["transformer"][f"layer{idx}"]["self_attention"]["attention"]["o_proj"][
                "weight"
            ].reshape(hidden_size, -1)
        )

        # convert normalization layers
        llama_state[f"model.layers.{idx}.input_layernorm.weight"] = jax_to_torch(
            state["decoder"]["transformer"][f"layer{idx}"]["self_attention"]["norm"]["scale"]
        )
        llama_state[f"model.layers.{idx}.post_attention_layernorm.weight"] = jax_to_torch(
            state["decoder"]["transformer"][f"layer{idx}"]["feed_forward"]["norm"]["scale"]
        )

    # convert normalization layer
    llama_state["model.norm.weight"] = jax_to_torch(state["decoder"]["output_norm"]["scale"])

    return as_torch_tensor(llama_state)


def get_fuji_and_llama(
    fuji_model_name,
    llama_model_name,
    load_true_model=False,
    reverse=False,
    fuji_model_path=None,
    llama_model_path=None,
):
    """Get fuji and llama with configuration matched.

    By default, fuji model will be modified to match llama.
    If reverse is set to True, Llama will be modified to match fuji."""
    llama_model_path = llama_model_path or llama_model_name

    # Llama-2-7b-hf vs fuji-7B-v2
    trainer_config = get_trainer_config(fuji_model_name)
    model_config = trainer_config.model
    model_config.set(name="model")

    if reverse:
        # initialize fuji model
        if load_true_model:
            if fuji_model_path is None:
                raise Exception("fuji_model_path not provided!")

            infer_runner, infer_runner_config = init_infer_runner(trainer_config, fuji_model_path)
            fuji = infer_runner.model
            state = infer_runner._inference_runner_state.model
        else:
            fuji = model_config.instantiate(parent=None)
            prng_key = jax.random.PRNGKey(0)
            # this does not take care of mesh, so will only work with reduced layer numbers
            # for large models like full 70B, generate a checkpoint first then use inference runner to take care of the mesh
            state = fuji.initialize_parameters_recursively(prng_key=prng_key)

        # initialize llama model
        config = AutoConfig.from_pretrained(
            f"{llama_model_name}_config.json",
            local_files_only=True,
        )
        config.num_hidden_layers = model_config.decoder.transformer.num_layers
        config.vocab_size = model_config.decoder.vocab_size
        config.eos_token_id = model_config.decoder.eos_token_id
        config.bos_token_id = -1
        llama = LlamaForCausalLM._from_config(config)
    else:
        # initialize transformer model
        if load_true_model:
            llama = LlamaForCausalLM.from_pretrained(llama_model_name, local_files_only=True)
        else:
            # self-specify smaller config for easier validation
            config = AutoConfig.from_pretrained(
                f"{llama_model_name}_config.json",
                local_files_only=True,
            )
            llama = LlamaForCausalLM._from_config(config)
        llama = llama.eval()

        # adjust num_layers to match the value in {llama_model_name}_config.json
        model_config.decoder.transformer.set(num_layers=llama.config.num_hidden_layers)
        # fuji model has different vocab size even for the same model size
        model_config.decoder.set(vocab_size=llama.config.vocab_size)

        if fuji_model_name == "fuji-7B-v2":
            # llama2 7B does not share lm_head with embedding, but fuji does
            # need to disable lm_head sharing for fuji to match llama
            model_config.decoder.set(lm_head=LmHead.default_config())

        # initialize fuji model
        fuji = model_config.instantiate(parent=None)
        prng_key = jax.random.PRNGKey(0)
        state = fuji.initialize_parameters_recursively(prng_key=prng_key)

    return fuji, state, llama


def generate_random_init_checkpoint(fuji_model_name, checkpoint_path):
    prng_key = jax.random.PRNGKey(0)
    trainer_config = get_trainer_config(fuji_model_name)
    model_config = trainer_config.model
    model_config.set(name="model")
    fuji = model_config.instantiate(parent=None)
    state = fuji.initialize_parameters_recursively(prng_key=prng_key)
    save_axlearn_checkpoint(fuji, state, checkpoint_path, get_mesh(trainer_config))


def validate_weights(fuji_model_name, llama_model_name, load_true_model=False, reverse=False):
    """Validate the weights after converting from llama to axlearn, then from axlearn back to llama is still the same."""
    fuji, state, llama = get_fuji_and_llama(
        fuji_model_name, llama_model_name, load_true_model, reverse
    )
    # conversion for llama2 and llama3 would be different
    # for example llama3 would use GQA and some of the model also share weights between lm_head and emb
    if llama_model_name in ["Llama-2-7b", "Llama-2-7b-hf"]:
        use_gqa = False
    else:
        use_gqa = True

    original_state_dict = deepcopy(llama.state_dict())
    state = parameters_from_llama(llama, state, use_gqa)
    llama_state_dict = parameters_to_llama(state, llama, use_gqa)

    for layer_name, layer_weights in original_state_dict.items():
        assert jnp.array_equal(layer_weights, llama_state_dict[layer_name])
    print("Validation finished successfully.")


def test_rope_conversion():
    tensor = torch.rand(4096, 32, 128)
    tensor2 = transformers_rope_to_axlearn_rope(tensor)
    tensor3 = axlearn_rope_to_transformers_rope(tensor2)
    assert jnp.array_equal(tensor, tensor2)
    assert jnp.array_equal(tensor, tensor3)


def run_all_tests():
    test_rope_conversion()
    validate_weights("fuji-7B-v2", "Llama-2-7b-hf")
    validate_weights("fuji-7B-v2", "Llama-2-7b-hf", load_true_model=True)
    validate_weights("fuji-7B-v2", "Llama-2-7b-hf", reverse=True)


if __name__ == "__main__":
    # validate_weights("fuji-7B-v2", "Llama-2-7b-hf")
    # validate_weights("fuji-7B-v2", "Llama-2-7b-hf", load_true_model=True)
    # validate_weights("fuji-7B-v2", "Llama-2-7b-hf", reverse=True)
    # validate_weights("fuji-1B-v3", "Llama-3.2-1B")
    # copy_files("Llama-2-7b-hf", "/fsx/czhenguo/Projects/fruitstand/runs/artifacts/axlearn_to_transformers/baseline_34000/")
    generate_random_init_checkpoint(
        "fuji-70B-v2",
        "/fsx/czhenguo/Projects/fruitstand/runs/artifacts/axlearn_venv/validation/fuji-70B-v2-full",
    )
