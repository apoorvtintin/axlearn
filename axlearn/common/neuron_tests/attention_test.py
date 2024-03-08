import contextlib

# pylint: disable=too-many-lines,duplicate-code,no-self-use

import jax
import pytest
import numpy as np
import optax
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from jax.sharding import NamedSharding
from axlearn.common import attention, test_utils, utils, causal_lm, optimizers
from axlearn.common.attention import (
    ParallelTransformerLayer,
    TransformerLayer, scaled_hidden_dim, TransformerFeedForwardLayer, MultiheadAttention, FusedQKVLinear, QKVLinear,
    StackedTransformerLayer, RepeatedTransformerLayer, PipelinedTransformerLayer, build_remat_spec,
)
from axlearn.common.base_layer import ParameterSpec, RematSpec
from axlearn.common.causal_lm import residual_initializer_cfg, TransformerStackConfig
from axlearn.common.config import config_for_function
from axlearn.common.decoder import Decoder, LmHead
from axlearn.common.embedding import TransformerTextEmbeddings
from axlearn.common.layers import RMSNorm, set_bias_recursively
from axlearn.common.learner import Learner
from axlearn.common.module import functional as F, InvocationContext, new_output_collection, set_current_context
from axlearn.common.optimizer_base import NestedOptParam, OptParam
from axlearn.common.optimizers import AddDecayedWeightsState
from axlearn.common.test_utils import NeuronTestCase, assert_allclose, dummy_segments_positions
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

from axlearn.common.utils import Tensor, VDict, NestedTensor, TensorSpec
import os

class TransformerTest(NeuronTestCase):
    """Tests TransformerLayer."""

    @pytest.mark.skip
    def test_forward_fused_qkv(self):
        """A test of TransformerLayer forward."""
        mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(4, 8)[:, None, None, None, :],
                                 axis_names=("data", "seq", "expert", "fsdp", "model"),)
        with mesh:
            model_dim = 512
            num_heads = 32
            cfg = TransformerLayer.default_config().set(name="test", input_dim=model_dim)
            cfg.feed_forward.set(hidden_dim=scaled_hidden_dim(4))
            cfg.self_attention.attention.set(num_heads=num_heads)
            cfg.self_attention.attention.input_linear = FusedQKVLinear.default_config()
            cfg.self_attention.norm = RMSNorm.default_config()
            cfg.feed_forward.norm = RMSNorm.default_config()
            set_bias_recursively(cfg, bias=False)
            set_double_shard_weights_config(
                cfg,
                batch_axis_names='data',
                fsdp_axis_names='fsdp',
                tp_axis_names='model',
                seq_axis_names='model',
            )
            layer: TransformerLayer = cfg.instantiate(parent=None)
            self._trainer_state_specs = collect_param_specs(layer)
            def create_named_sharding(param_spec, mesh):
                if isinstance(param_spec, ParameterSpec):
                    return NamedSharding(
                        mesh,
                        PartitionSpec(*param_spec.mesh_axes) if param_spec.mesh_axes != (None,) else PartitionSpec(None)
                    )
                return param_spec

            def custom_tree_map(func, pytree, mesh):
                if isinstance(pytree, dict) or isinstance(pytree, VDict):
                    new_dict = {}
                    for k, v in pytree.items():
                        if k == 'i_proj': # Weird case where i_proj is a Vdict not a Dict
                            new_dict[k] = VDict({sub_k: custom_tree_map(func, sub_v, mesh) for sub_k, sub_v in v.items()})
                        else:
                            new_dict[k] = custom_tree_map(func, v, mesh)
                    return type(pytree)(new_dict)
                else:
                    return func(pytree)

            self._trainer_state_partition_specs = custom_tree_map(
                lambda ps: create_named_sharding(ps, mesh),
                self._trainer_state_specs,
                mesh
            )
            def init_cpu():  # Initing on Neuron causes compiler failures.
                layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(0))
                return layer_params

            def move_to_neuron(params):
                weights = jax.device_put(params)
                return weights
            cpu_device = jax.devices('cpu')[0]
            with jax.default_device(cpu_device):
                layer_params = init_cpu()

            move_to_neuron = jax.jit(
                move_to_neuron,
                in_shardings=(self._trainer_state_partition_specs,), # tuple is necessary here
            )
            layer_params = move_to_neuron(layer_params)
            def print_dict_structure(d, indent=0):
                for key, value in d.items():
                    print(' ' * indent + f"{key}: {type(value)}")
                    if isinstance(value, dict):
                        print_dict_structure(value, indent + 4)

            #print_dict_structure(layer_params)

            jax.debug.visualize_array_sharding(layer_params['feed_forward']['linear1']['weight'])
            layer.self_attention.norm = jax.jit(layer.self_attention.norm, in_shardings=(NamedSharding(mesh, PartitionSpec('data', 'model', None)),),
                                                out_shardings=(NamedSharding(mesh, PartitionSpec('data', None, None))))
            batch_size, tgt_len = 4, 256
            rng = np.random.default_rng(seed=123)
            target = rng.random([batch_size, tgt_len, model_dim], dtype=np.float32)
            mask = attention.make_causal_mask(tgt_len)
            mask = jnp.tile(mask[None, None, :, :], (batch_size, num_heads, 1, 1))
            mask = jax.device_put(mask, NamedSharding(mesh, PartitionSpec('data', 'model', None, None)))
            input_tensor = jnp.asarray(target)
            input_tensor = jax.device_put(input_tensor, NamedSharding(mesh, PartitionSpec('data', 'model', None)))

            def run(mask, input_tensor, weights):
                layer_outputs, _ = F(
                    layer,
                    inputs=dict(data=input_tensor, self_attention_logit_biases=mask),
                    state=weights,
                    is_training=True,
                    prng_key=jax.random.PRNGKey(0),
                )
                return layer_outputs
            run = jax.jit(run, in_shardings=(NamedSharding(mesh, PartitionSpec('data', 'model', None, None)),
                                             NamedSharding(mesh, PartitionSpec('data', 'model', None)),
                                             self._trainer_state_partition_specs))
            layer_outputs = run(mask, input_tensor, layer_params)
            self.assertEqual(target.shape, layer_outputs.data.shape)
    @pytest.mark.skip
    def test_backward_fused_qkv(self):
        """A test of TransformerLayer backward."""
        mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(4, 8)[:, None, None, None, :],
                                 axis_names=("data", "seq", "expert", "fsdp", "model"),)
        with mesh:
            model_dim = 4096
            num_heads = 32
            cfg = TransformerLayer.default_config().set(name="test", input_dim=model_dim)
            cfg.dtype = jnp.bfloat16
            cfg.feed_forward.set(hidden_dim=scaled_hidden_dim(4))
            cfg.self_attention.attention.set(num_heads=num_heads)
            cfg.self_attention.attention.input_linear = FusedQKVLinear.default_config()
            cfg.self_attention.norm = RMSNorm.default_config()
            cfg.feed_forward.norm = RMSNorm.default_config()
            set_bias_recursively(cfg, bias=False)
            set_double_shard_weights_config(
                cfg,
                batch_axis_names='data',
                fsdp_axis_names='fsdp',
                tp_axis_names='model',
                seq_axis_names='model',
            )
            layer: TransformerLayer = cfg.instantiate(parent=None)
            self._trainer_state_specs = collect_param_specs(layer)
            def create_named_sharding(param_spec, mesh):
                if isinstance(param_spec, ParameterSpec):
                    return NamedSharding(
                        mesh,
                        PartitionSpec(*param_spec.mesh_axes) if param_spec.mesh_axes != (None,) else PartitionSpec(None)
                    )
                return param_spec

            def custom_tree_map(func, pytree, mesh):
                if isinstance(pytree, dict) or isinstance(pytree, VDict):
                    new_dict = {}
                    for k, v in pytree.items():
                        if k == 'i_proj': # Weird case where i_proj is a Vdict not a Dict
                            new_dict[k] = VDict({sub_k: custom_tree_map(func, sub_v, mesh) for sub_k, sub_v in v.items()})
                        else:
                            new_dict[k] = custom_tree_map(func, v, mesh)
                    return type(pytree)(new_dict)
                else:
                    return func(pytree)

            self._trainer_state_partition_specs = custom_tree_map(
                lambda ps: create_named_sharding(ps, mesh),
                self._trainer_state_specs,
                mesh
            )
            def init_cpu():  # Initing on Neuron causes compiler failures.
                layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(0))
                return layer_params

            def move_to_neuron(params):
                weights = jax.device_put(params)
                return weights
            cpu_device = jax.devices('cpu')[0]
            with jax.default_device(cpu_device):
                layer_params = init_cpu()

            move_to_neuron = jax.jit(
                move_to_neuron,
                in_shardings=(self._trainer_state_partition_specs,), # singleton tuple is necessary here
            )
            layer_params = move_to_neuron(layer_params)
            def print_dict_structure(d, indent=0):
                for key, value in d.items():
                    print(' ' * indent + f"{key}: {type(value)}")
                    if isinstance(value, dict):
                        print_dict_structure(value, indent + 4)

            #print_dict_structure(layer_params)

            jax.debug.visualize_array_sharding(layer_params['feed_forward']['linear1']['weight'])
            #layer.self_attention.norm = jax.jit(layer.self_attention.norm, in_shardings=(NamedSharding(mesh, PartitionSpec('data', 'model', None)),),
                                      #out_shardings=(NamedSharding(mesh, PartitionSpec('data', None, None))))
            # Above jit will prevent an all to all.
            batch_size, tgt_len = 4, 4096
            rng = np.random.default_rng(seed=123)
            target = rng.random([batch_size, tgt_len, model_dim], dtype=np.float32).astype(jnp.bfloat16)
            target = jax.device_put(target, NamedSharding(mesh, PartitionSpec('data', None, None)))
            def mask_creation():
                mask = attention.make_causal_mask(tgt_len).astype(jnp.bfloat16)
                mask = jnp.tile(mask[None, None, :, :], (batch_size, num_heads, 1, 1))
                mask = jax.device_put(mask)
                return mask
            mask_creation = jax.jit(mask_creation, out_shardings=NamedSharding(mesh, PartitionSpec('data', 'model', None, None)))
            mask = mask_creation()
            input_tensor = jnp.asarray(target).astype(jnp.bfloat16)
            #input_tensor = jax.device_put(input_tensor, NamedSharding(mesh, PartitionSpec('data', 'model', None)))
            input_tensor = jax.device_put(input_tensor, NamedSharding(mesh, PartitionSpec('data', None, None)))

            def run(mask, input_tensor, output_target, weights):
                layer_outputs, _ = F(
                    layer,
                    inputs=dict(data=input_tensor, self_attention_logit_biases=mask),
                    state=weights,
                    is_training=True,
                    prng_key=jax.random.PRNGKey(0),
                )

                return jnp.mean((layer_outputs.data - output_target) ** 2)

            run = jax.jit(jax.value_and_grad(run), in_shardings=(NamedSharding(mesh, PartitionSpec('data', 'model', None, None)),
                                             NamedSharding(mesh, PartitionSpec('data', None, None)),
                                             NamedSharding(mesh, PartitionSpec('data', None, None)),
                                             self._trainer_state_partition_specs))
            loss, grad = run(mask, input_tensor, target, layer_params)
            print(loss)
            print(grad)

    def test_model(self):
        """A test of Stacked TransformerLayer backward."""
        TP_DEGREE = 8
        DP_DEGREE = (int(os.getenv('SLURM_JOB_NUM_NODES'))*32)//TP_DEGREE
        mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(DP_DEGREE, TP_DEGREE)[:, None, None, None, :],
                                 axis_names=("data", "seq", "expert", "fsdp", "model"),)
        with mesh:
            model_dim = 4096
            num_heads = 32
            vocab_size = 32000
            stacked_layer = StackedTransformerLayer.default_config()
            decoder_cfg = llama_decoder_config(
                stack_cfg=stacked_layer,
                num_layers=4,
                hidden_dim=model_dim,
                num_heads=num_heads,
                vocab_size=vocab_size,
                activation_function="nn.gelu",
                layer_norm_epsilon=0.1,
                dropout_rate=0.0,
            )
            model_cfg = causal_lm.Model.default_config().set(decoder=decoder_cfg, name="llama")
            #print(model_cfg)
            set_model_shard_weights_config(
                model_cfg,
                batch_axis_names='data',
                fsdp_axis_names='fsdp',
                tp_axis_names='model',
                seq_axis_names='model',
            )
            model = model_cfg.instantiate(parent=None)

            adamw = config_for_function(optimizers.adamw_optimizer).set(
                learning_rate=1e-4, b1=0.9, b2=0.95, eps=1e-6, mu_dtype=jnp.float32
            )
            learn_cfg = Learner.default_config().set(
                name="test", optimizer=adamw)
            learner: Learner = learn_cfg.instantiate(parent=None)


            self._trainer_state_specs = collect_param_specs(model)
            def create_named_sharding(param_spec, mesh):
                if isinstance(param_spec, ParameterSpec):
                    return NamedSharding(
                        mesh,
                        PartitionSpec(*param_spec.mesh_axes) if param_spec.mesh_axes != (None,) else PartitionSpec(None)
                    )
                return param_spec

            def custom_tree_map(func, pytree, mesh):
                if isinstance(pytree, dict) or isinstance(pytree, VDict):
                    new_dict = {}
                    for k, v in pytree.items():
                        if k == 'i_proj': # Weird case where i_proj is a Vdict not a Dict
                            new_dict[k] = VDict({sub_k: custom_tree_map(func, sub_v, mesh) for sub_k, sub_v in v.items()})
                        else:
                            new_dict[k] = custom_tree_map(func, v, mesh)
                    return type(pytree)(new_dict)
                else:
                    return func(pytree)

            self._trainer_state_partition_specs = custom_tree_map(
                lambda ps: create_named_sharding(ps, mesh),
                self._trainer_state_specs,
                mesh
            )
            #print(f'Trainer State Specs = {self._trainer_state_specs}')
            #print(f'Trainer State Partition Specs = {self._trainer_state_partition_specs}')
            learner_state_specs = learner.create_state_partition_specs(
                model.create_parameter_specs_recursively()
            )
            #print(f'Learner State Specs = {learner_state_specs}')
            zero1 = True
            def create_named_sharding_optimizer(tensor_spec, mesh):
                print(f"Creating NamedSharding for: {tensor_spec}")
                if isinstance(tensor_spec, TensorSpec):
                    # Check if mesh_axes is a single element tuple with None, then use PartitionSpec(None)
                    if tensor_spec.mesh_axes == (None,):
                        print(f"  Using PartitionSpec(None) for tensor spec: {tensor_spec}")
                        return NamedSharding(mesh, PartitionSpec(None))
                    else:
                        # Check if the number of axes in mesh_axes exceeds the tensor's dimensions
                        if len(tensor_spec.mesh_axes) > len(tensor_spec.shape):
                            # Pop off the first axis
                            adjusted_mesh_axes = tensor_spec.mesh_axes[1:]
                            print(f"  Adjusting mesh_axes from {tensor_spec.mesh_axes} to {adjusted_mesh_axes}")
                        else:
                            adjusted_mesh_axes = tensor_spec.mesh_axes

                        # Convert adjusted_mesh_axes to PartitionSpec
                        if zero1:
                            adjusted_mesh_axes = tuple('data' if axis == 'fsdp' else axis for axis in adjusted_mesh_axes)
                        partition_spec = PartitionSpec(*adjusted_mesh_axes)
                        print(f"  Using PartitionSpec({partition_spec}) for tensor spec: {tensor_spec}")
                        return NamedSharding(mesh, partition_spec)
                # If it's not a TensorSpec, return as is
                return tensor_spec

            def convert_specs(specs, mesh):
                # Handle tuples by processing each element
                if isinstance(specs, optax.ScaleByAdamState):
                    #print(f"Converting ScaleByAdamState with count TensorSpec: {specs.count}")
                    converted_count = create_named_sharding_optimizer(specs.count, mesh)
                    return optax.ScaleByAdamState(
                        count=None,
                        mu=convert_specs(specs.mu, mesh),
                        nu=convert_specs(specs.nu, mesh)
                    )
                elif isinstance(specs, AddDecayedWeightsState):
                    #print(f"Converting AddDecayedWeightsState with count: {specs.count}")
                    #if specs.count is not None:
                    converted_count = create_named_sharding_optimizer(specs.count, mesh)
                    #else:
                     #   converted_count = None
                    return AddDecayedWeightsState(count=None)
                elif isinstance(specs, optax.ScaleByScheduleState):
                    #print(f"Converting ScaleByScheduleState with count TensorSpec: {specs.count}")
                    converted_count = create_named_sharding_optimizer(specs.count, mesh)
                    return optax.ScaleByScheduleState(count=None)
                elif isinstance(specs, TensorSpec):
                    #print(f"Converting TensorSpec with shape {specs.shape} and dtype {specs.dtype}")
                    return create_named_sharding_optimizer(specs, mesh)
                elif isinstance(specs, tuple):
                    #print(f"Converting tuple with elements: {specs}")
                    return tuple(convert_specs(spec, mesh) for spec in specs)
                elif isinstance(specs, dict):
                    #print(f"Converting dictionary with keys: {list(specs.keys())}")
                    if isinstance(specs, VDict):
                        return VDict({key: convert_specs(value, mesh) for key, value in specs.items()})
                    return {key: convert_specs(value, mesh) for key, value in specs.items()}
                else:
                    #print(f"Returning unrecognized spec type as is: {type(specs)}")
                    return specs

            self._learner_state_partition_specs = convert_specs(learner_state_specs, mesh)
            print(f' Learner State Partition Specs = {self._learner_state_partition_specs}')

            def init_cpu():  # Initing on Neuron causes compiler failures.
                model_params = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(0))
                _model_param_specs = model.create_parameter_specs_recursively()
                learner_params = learner.init(_opt_params(model_params, _model_param_specs))
                return model_params, learner_params

            def move_to_neuron(model_params, learner_params):
                model_weights = jax.device_put(model_params)
                learner_weights = jax.device_put(learner_params)
                return model_weights, learner_weights
            cpu_device = jax.devices('cpu')[0]
            with jax.default_device(cpu_device):
                model_params, learner_params = init_cpu()

            move_to_neuron = jax.jit(
                move_to_neuron,
                in_shardings=(self._trainer_state_partition_specs, self._learner_state_partition_specs), # singleton tuple is necessary here
            )
            model_params, learner_params = move_to_neuron(model_params, learner_params)
            def print_dict_structure(d, indent=0):
                for key, value in d.items():
                    print(' ' * indent + f"{key}: {type(value)}")
                    if isinstance(value, dict):
                        print_dict_structure(value, indent + 4)

            #print_dict_structure(model_params)
            #print_dict_structure(learner_params)
            print(learner_params)

            print(self._learner_state_partition_specs)
            jax.debug.visualize_array_sharding(model_params['decoder']['transformer']['layer0']['feed_forward']['linear1']['weight'])
            #norm = jax.jit(model.decoder.transformer.layer0.self_attention.norm, in_shardings=(NamedSharding(mesh, PartitionSpec('data', 'model', None)),),
             #              out_shardings=(NamedSharding(mesh, PartitionSpec('data', None, None))))
            #create_causal_mask = jax.jit(model.decoder.create_causal_mask, out_shardings=NamedSharding(mesh, PartitionSpec('data', 'model', None, None)))
            #model.decoder.create_causal_mask = create_causal_mask
            #model.decoder.output_norm = norm
            #for layer in model.decoder.transformer._layers:
            #    layer.norm = norm
            #    layer.self_attention.norm = norm
            #    layer.feed_forward.norm = norm

            # in_shardings=(NamedSharding(mesh, PartitionSpec('data', None, None)),
            # (NamedSharding(mesh, PartitionSpec('data', 'model', None, None)))),
            #self_attention = jax.jit(model.decoder.transformer.layer0.self_attention.attention,
             #                            out_shardings=(NamedSharding(mesh, PartitionSpec('data', None, None)))
              #                           )  # this line fixes all to all in backwards but breaks Neuron compiler
            #for layer in model.decoder.transformer._layers:
            #    layer.self_attention.attention = self_attention

            # Above jit will prevent an all to all.
            batch_size, tgt_len = DP_DEGREE, 4096
            seq_len = 2048 # the individual seq length packed into 4096
            rng = np.random.default_rng(seed=123)

            input_ids = jax.random.randint(
                jax.random.PRNGKey(123), shape=[batch_size, tgt_len], minval=0, maxval=vocab_size-2
            )
            target_labels = jax.random.randint(
                jax.random.PRNGKey(123), shape=[batch_size, tgt_len], minval=0, maxval=vocab_size-2
            )
            global_shape = (batch_size, tgt_len)
            sharding = jax.sharding.NamedSharding(mesh, PartitionSpec('data', None))

            arrays = [
                jax.device_put(input_ids[index], d)
                for d, index in sharding.addressable_devices_indices_map(global_shape).items()]
            input_ids = jax.make_array_from_single_device_arrays(global_shape, sharding, arrays)
            arrays_target = [
                jax.device_put(target_labels[index], d)
                for d, index in sharding.addressable_devices_indices_map(global_shape).items()]

            target_labels = jax.make_array_from_single_device_arrays(global_shape, sharding, arrays_target)

            segment_ids = jnp.zeros((batch_size, tgt_len), dtype=jnp.int32)
            segment_ids = segment_ids.at[:, seq_len:].set(1)
            positions = jnp.tile(jnp.arange(seq_len), (batch_size, 2))
            segment_ids_arrays = [
                jax.device_put(segment_ids[index], d)
                for d, index in sharding.addressable_devices_indices_map((batch_size, tgt_len)).items()
            ]
            segment_ids = jax.make_array_from_single_device_arrays((batch_size, tgt_len), sharding, segment_ids_arrays)

            positions_arrays = [
                jax.device_put(positions[index], d)
                for d, index in sharding.addressable_devices_indices_map((batch_size, tgt_len)).items()
            ]
            positions = jax.make_array_from_single_device_arrays((batch_size, tgt_len), sharding, positions_arrays)

            def run(input_ids, target_labels, segment_ids, position_ids, model_params):
                ctx = InvocationContext(
                    name="root",
                    parent=None,
                    module=model,
                    state=model_params,
                    output_collection=new_output_collection(),
                    is_training=True,
                    prng_key=jax.random.PRNGKey(123),
                )
                with set_current_context(ctx):
                    input_batch = dict(input_ids=input_ids, target_labels=target_labels,
                                       input_segment_ids=segment_ids, positions=position_ids)
                    loss = model.forward(input_batch=input_batch, return_aux=False)
                return loss[0]
            # ptoulme differentiate with respect to argnums=4 the weights
            run = jax.jit(jax.value_and_grad(run, argnums=4), in_shardings=(NamedSharding(mesh, PartitionSpec('data', None)),
                                                                 NamedSharding(mesh, PartitionSpec('data', None)),
                                                                 NamedSharding(mesh, PartitionSpec('data', None)),
                                                                 NamedSharding(mesh, PartitionSpec('data', None)),
                                                                 self._trainer_state_partition_specs))

            def train_step(input_ids, target_labels, segment_ids, positions, model_params, learner_params):
                loss, grad = run(input_ids, target_labels, segment_ids, positions, model_params)
                def optimizer_step(state, grads, params):
                    updated_params, output_collection = F(
                            learner,
                            method="update",
                            is_training=True,
                            prng_key=jax.random.PRNGKey(123),
                            state=state,
                            inputs=dict(
                                gradients=grads,
                                model_params=params,
                                state_updates={},
                            ),
                    )
                    return updated_params
                _model_param_specs = model.create_parameter_specs_recursively()
                opt_params = _opt_params(model_params, _model_param_specs)

                updated_params = optimizer_step(learner_params, grad, opt_params)
                return loss, grad, updated_params

            train_step = jax.jit(train_step, in_shardings=(NamedSharding(mesh, PartitionSpec('data', None)),
                                                          NamedSharding(mesh, PartitionSpec('data', None)),
                                                          NamedSharding(mesh, PartitionSpec('data', None)),
                                                          NamedSharding(mesh, PartitionSpec('data', None)),
                                                          self._trainer_state_partition_specs,
                                                          self._learner_state_partition_specs))
            loss, grad, weights = train_step(input_ids, target_labels, segment_ids, positions, model_params, learner_params)
            print(f'Loss={loss}')
            print(f'Grad={grad}')
            print(f'Weight={weights}')
def set_double_shard_weights_config(
        cfg: Union[TransformerLayer.Config, Sequence[TransformerLayer.Config]],
        *,
        batch_axis_names: Union[str, Sequence[str]] = ("data", "fsdp"),
        fsdp_axis_names: Union[str, Sequence[str]] = "fsdp",
        tp_axis_names: Union[str, Sequence[str]] = "model",
        seq_axis_names: Union[str, Sequence[str]] = "seq",
):
    """Sets `cfg` to shard FFN and attention weights over both fsdp and tp axes.

    Args:
        cfg: (A sequence of) Transformer layer config to apply sharding spec to.
        batch_axis_names: Axis name(s) over which we shard the batch dimension of output tensors.
        fsdp_axis_names: Axis name(s) over which we shard fully-sharded-data-parallel tensors.
        tp_axis_names: Axis name(s) over which we shard tensor-parallel tensors.
        seq_axis_names: Axis name(s) over which we shard sequence-parallel tensors.
    """

    # pytype: disable=attribute-error
    def set_attn_partition_specs(attn_layer: MultiheadAttention.Config):
        # Shard weights.
        input_linear_cfg = attn_layer.input_linear
        if hasattr(input_linear_cfg, "input_linear"):
            input_linear_cfg = input_linear_cfg.input_linear
        input_linear_cfg.layer.param_partition_spec = (None, fsdp_axis_names, tp_axis_names, None)
        # ptoulme bug - when FusedQKV is enabled it has a shape (3, hidden, num_heads, head_dimension) dimension so add a (None to account for this
        attn_layer.output_linear.param_partition_spec = (fsdp_axis_names, tp_axis_names, None)

    def set_ffn_partition_specs(ff_layer: TransformerFeedForwardLayer.Config):
        # Shard weights.
        ff_layer.linear1.param_partition_spec = (fsdp_axis_names, tp_axis_names)
        ff_layer.linear2.param_partition_spec = (tp_axis_names, fsdp_axis_names)
        # Encourage the right activation sharding.
        ff_layer.linear1.output_partition_spec = (batch_axis_names, None, tp_axis_names)
        ff_layer.linear2.output_partition_spec = (batch_axis_names, seq_axis_names, None)

    if not isinstance(cfg, Sequence):
        cfg = [cfg]

    for layer_cfg in cfg:
        set_attn_partition_specs(layer_cfg.self_attention.attention)
        if layer_cfg.cross_attention is not None:
            set_attn_partition_specs(layer_cfg.cross_attention.attention)
        if isinstance(layer_cfg.feed_forward, TransformerFeedForwardLayer.Config):
            set_ffn_partition_specs(layer_cfg.feed_forward)


def collect_param_specs(layer):
    # If the layer has children, recurse into them and collect their specs.
    if hasattr(layer, '_children') and layer._children:
        return {name: collect_param_specs(child) for name, child in layer._children.items()}
    else:
        # Otherwise, return the parameter specs of the current layer.
        return layer._create_layer_parameter_specs()



def llama_decoder_config(
        stack_cfg: TransformerStackConfig,
        num_layers: int,
        hidden_dim: int,
        num_heads: int,
        vocab_size: int,
        activation_function: str = "nn.relu",
        layer_norm_epsilon: float = 1e-08,
        dropout_rate: float = 0.0,
        layer_remat: Optional[RematSpec] = None,
) -> Decoder.Config:
    """Build a decoder transformer config in the style of GPT.

    Reference: https://github.com/openai/gpt-2.

    Args:
        stack_cfg: A config of StackedTransformerLayer, RepeatedTransformerLayer, or
            PipelinedTransformerLayer.
        num_layers: Number of transformer decoder layers.
        hidden_dim: Dimension of embeddings and input/output of each transformer layer.
        num_heads: Number of attention heads per transformer layer.
        vocab_size: Size of vocabulary.
        max_position_embeddings: Number of positional embeddings.
        activation_function: Type of activation function.
        layer_norm_epsilon: Epsilon for layer normalization. Defaults to LayerNorm.config.eps.
        dropout_rate: Dropout rate applied throughout model, including output_dropout.
        layer_remat: If not None, use as transformer.layer.remat_spec.

    Returns:
        A Decoder config.
    """
    stack_cfg = stack_cfg.clone()

    assert stack_cfg.klass in [
        StackedTransformerLayer,
        RepeatedTransformerLayer,
        PipelinedTransformerLayer,
    ]

    cfg = TransformerLayer.default_config()
    cfg.dtype = jnp.bfloat16
    #cfg.dtype = jnp.float32
    cfg.feed_forward.set(hidden_dim=scaled_hidden_dim(4))
    cfg.self_attention.attention.set(num_heads=num_heads)
    cfg.self_attention.attention.input_linear = FusedQKVLinear.default_config()
    cfg.self_attention.norm = RMSNorm.default_config()
    cfg.feed_forward.norm = RMSNorm.default_config()
    set_bias_recursively(cfg, bias=False)

    transformer_cls = stack_cfg.set(num_layers=num_layers, layer=cfg)
    decoder = Decoder.default_config().set(
        transformer=transformer_cls,
        dim=hidden_dim,
        vocab_size=vocab_size,
        emb=TransformerTextEmbeddings.default_config().set(pos_emb=None).set(dtype=jnp.bfloat16), #bfloat16
        output_norm=RMSNorm.default_config().set(eps=layer_norm_epsilon),
        dropout_rate=dropout_rate,
        lm_head=LmHead.default_config().set(dtype=jnp.bfloat16)  #bfloat16
    )
    return decoder




def set_model_shard_weights_config(
        cfg: Union[TransformerLayer.Config, Sequence[TransformerLayer.Config]],
        *,
        batch_axis_names: Union[str, Sequence[str]] = ("data", "fsdp"),
        fsdp_axis_names: Union[str, Sequence[str]] = "fsdp",
        tp_axis_names: Union[str, Sequence[str]] = "model",
        seq_axis_names: Union[str, Sequence[str]] = "seq",
):
    """Sets `cfg` to shard FFN and attention weights over both fsdp and tp axes.

    Args:
        cfg: (A sequence of) Transformer layer config to apply sharding spec to.
        batch_axis_names: Axis name(s) over which we shard the batch dimension of output tensors.
        fsdp_axis_names: Axis name(s) over which we shard fully-sharded-data-parallel tensors.
        tp_axis_names: Axis name(s) over which we shard tensor-parallel tensors.
        seq_axis_names: Axis name(s) over which we shard sequence-parallel tensors.
    """

    # pytype: disable=attribute-error
    def set_attn_partition_specs(attn_layer: MultiheadAttention.Config):
        # Shard weights.
        input_linear_cfg = attn_layer.input_linear
        if hasattr(input_linear_cfg, "input_linear"):
            input_linear_cfg = input_linear_cfg.input_linear
        input_linear_cfg.layer.param_partition_spec = (None, fsdp_axis_names, tp_axis_names, None)
        # ptoulme bug - when FusedQKV is enabled it has a shape (3, hidden, num_heads, head_dimension) dimension so add a (None to account for this
        attn_layer.output_linear.param_partition_spec = (fsdp_axis_names, tp_axis_names, None)

    def set_ffn_partition_specs(ff_layer: TransformerFeedForwardLayer.Config):
        # Shard weights.
        ff_layer.linear1.param_partition_spec = (fsdp_axis_names, tp_axis_names)
        ff_layer.linear2.param_partition_spec = (tp_axis_names, fsdp_axis_names)
        # Encourage the right activation sharding.
        ff_layer.linear1.output_partition_spec = (batch_axis_names, None, tp_axis_names)
        ff_layer.linear2.output_partition_spec = (batch_axis_names, None, None)

    #if not isinstance(cfg, Sequence):
     #   cfg = [cfg]
    #print(cfg.decoder)
    cfg.decoder.emb.token_emb.param_partition_spec = (fsdp_axis_names, tp_axis_names) # shard hidden
    cfg.decoder.lm_head.param_partition_spec = (tp_axis_names, fsdp_axis_names) # shard vocab
    for layer_cfg in [cfg.decoder.transformer.layer]: # shard the sole layer and its used for all other layers
        layer_cfg.remat_spec = build_remat_spec(cfg.decoder.transformer) # activation checkpointing
        set_attn_partition_specs(layer_cfg.self_attention.attention)
        if layer_cfg.cross_attention is not None:
            set_attn_partition_specs(layer_cfg.cross_attention.attention)
        if isinstance(layer_cfg.feed_forward, TransformerFeedForwardLayer.Config):
            set_ffn_partition_specs(layer_cfg.feed_forward)


# ptoulme - taken from Trainer
def _opt_params(model_params: NestedTensor, _model_param_specs) -> NestedOptParam:
    """Returns a tree of OptParam for Learner.{init,update}."""
    # self._model_param_specs can be incomplete. Complete it first.
    specs = utils.complete_partition_spec_tree(
        jax.tree_util.tree_structure(model_params), _model_param_specs
    )
    return jax.tree_util.tree_map(
        lambda param, spec: OptParam(
            value=param,
            factorization_spec=spec.factorization if spec is not None else None,
            weight_decay_scale=spec.weight_decay_scale if spec is not None else 1.0,
        ),
        model_params,
        specs,
    )