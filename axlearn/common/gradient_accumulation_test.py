# Copyright © 2024 Apple Inc.
"""Test module for gradient_accumulation.py"""
import chex
import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized
import pytest
from functools import partial
import numpy as np

from axlearn.common import gradient_accumulation, test_utils
from axlearn.common.metrics import MetricAccumulator, WeightedScalar
from axlearn.common.module import new_output_collection
from axlearn.common.update_transformation import ForwardOutputs
from axlearn.common.utils import Nested, PartitionSpec, Tensor, tree_paths
from axlearn.common.input_base import Input, PathAndRank, partition_by_path_rank
from jax.experimental.pjit import pjit
from typing import Callable, Union
from axlearn.common.config import ConfigOr, config_for_function, maybe_instantiate

def check_sharding(
    *,
    input_batch: Nested[Tensor],
    callback: Callable[[str, jax.sharding.Sharding], None],
):
    """Instantiates an input, dispatches the input batch, and invokes callback with the sharding.

    The callback is invoked with (path: str, sharding: Sharding).
    """

    def check_sharding(path, value):
        # print("checking sharding ", path, value)
        jax.debug.inspect_array_sharding(value, callback=lambda sharding: callback(path, sharding))

    jax.tree_map(check_sharding, tree_paths(input_batch), input_batch)
    return input_batch
class TestMinibatchPartitioner(test_utils.TestCase):
    """Test `with_minibatch_steps` decorator."""
    """Tests Input."""

    def create_dummy_params_batches(self, steps):
        self.batch_size = 4
        self.seq_len = 8
        self.params = dict(
            w=jnp.asarray([0.0, 2.0, 2.0, -3.0]),
            b=jnp.asarray([0.0, -1.0, 0.0, 0.0]),
        )

        self.batch_size *= steps
        self.input_batch = {
            "input_ids": jnp.ones((self.batch_size, self.seq_len), dtype=jnp.int32),
            "target_labels": jnp.ones((self.batch_size, self.seq_len), dtype=jnp.int32),
            "target_num_bytes": jnp.ones((self.batch_size,), dtype=jnp.int32),
        }
        forward_key, param_noise_key = jax.random.split(jax.random.PRNGKey(0), 2)
        self.inputs = dict(
            input_batch=self.input_batch,
            forward_key=forward_key,
            param_noise_key=param_noise_key,
        )

    def create_loss_fn(self, expected):
        def loss_fn(*, model_params, inputs) -> ForwardOutputs:
            """Simple ForwardFn."""
            self.check(
                input_batch=inputs["input_batch"],
                expected=expected,
            )
            loss = -jax.nn.log_softmax(model_params["w"] + model_params["b"])[1]
            output_collection = new_output_collection()
            output_collection.state_updates["w"] = model_params["w"] + 1
            output_collection.state_updates["loss"] = WeightedScalar(loss, 1)
            return ForwardOutputs(loss=loss, aux={}, output_collection=output_collection)
        return loss_fn

    @pytest.mark.skipif(
        jax.device_count() != 4 or jax.process_count() != 1,
        reason=(
            "Incorrect device & process count for mesh.\n"
            "Use XLA_FLAGS=--xla_force_host_platform_device_count=4 to run locally."
        ),
    )
    @parameterized.named_parameters(
        ("one_step", 1),  # no accumulation
        ("two_steps", 2),
        ("four_steps", 4),
    )
    def test_minibatch_partitioner_default(self, steps):
        """Tests grad accumulation with minibatch steps."""
        
        batch_size = 4 * steps
        seq_len = 8
        input_batch = {
            "input_ids": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
            "target_labels": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
            "target_num_bytes": jnp.ones((batch_size,), dtype=jnp.int32),
        }

        with jax.sharding.Mesh(
            np.array(jax.devices()).reshape(1, 2, 1, 2)[..., None],
            axis_names=("expert", "data", "fsdp", "seq", "model"),
        ):
            params = dict(
                w=jnp.asarray([0.0, 2.0, 2.0, -3.0]),
                b=jnp.asarray([0.0, -1.0, 0.0, 0.0]),
            )
            

            def check(input_batch: Nested[Tensor], expected: dict):
                callback = lambda path, sharding: self.assertEqual(expected[path], sharding.spec)
                
                check_sharding(
                    input_batch=input_batch,
                    callback=callback,
                )

            def loss_fn(*, model_params, inputs) -> ForwardOutputs:
                """Simple ForwardFn."""
                # Without input partitioner, constrain along batch axis names.
                check(
                    input_batch=inputs["input_batch"],
                    expected={
                        "input_ids": PartitionSpec(("data"), "seq"),
                        "target_labels": PartitionSpec(("data"), "seq"),
                        "target_num_bytes": PartitionSpec(("data")),
                    },
                )
                loss = -jax.nn.log_softmax(model_params["w"] + model_params["b"])[1]
                output_collection = new_output_collection()
                output_collection.state_updates["w"] = model_params["w"] + 1
                output_collection.state_updates["loss"] = WeightedScalar(loss, 1)
                return ForwardOutputs(loss=loss, aux={}, output_collection=output_collection)

            forward_key, param_noise_key = jax.random.split(jax.random.PRNGKey(0), 2)
            inputs = dict(
                input_batch=input_batch,
                forward_key=forward_key,
                param_noise_key=param_noise_key,
            )
            
            loss_fn = gradient_accumulation.with_minibatch_steps(
                steps=steps, metric_accumulator=MetricAccumulator.default_config(), minibatch_partitioner=None
            )(loss_fn)
            
            pjit(loss_fn, in_shardings=None).lower(model_params=params, inputs=inputs).compile()

    @pytest.mark.skipif(
        jax.device_count() != 4 or jax.process_count() != 1,
        reason=(
            "Incorrect device & process count for mesh.\n"
            "Use XLA_FLAGS=--xla_force_host_platform_device_count=4 to run locally."
        ),
    )
    @parameterized.named_parameters(
        ("one_step", 1),  # no accumulation
        ("two_steps", 2),
        ("four_steps", 4),
    )
    def test_minibatch_partitioner_non_default(self, steps):
        """Tests grad accumulation with minibatch steps."""
        
        batch_size = 4 * steps
        seq_len = 8
        input_batch = {
            "input_ids": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
            "target_labels": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
            "target_num_bytes": jnp.ones((batch_size,), dtype=jnp.int32),
        }

        with jax.sharding.Mesh(
            np.array(jax.devices()).reshape(1, 2, 1, 2)[..., None],
            axis_names=("expert", "data", "fsdp", "seq", "model"),
        ):
            params = dict(
                w=jnp.asarray([0.0, 2.0, 2.0, -3.0]),
                b=jnp.asarray([0.0, -1.0, 0.0, 0.0]),
            )
            

            def check(input_batch: Nested[Tensor], expected: dict):
                callback = lambda path, sharding: self.assertEqual(expected[path], sharding.spec)
                
                check_sharding(
                    input_batch=input_batch,
                    callback=callback,
                )

            def loss_fn(*, model_params, inputs) -> ForwardOutputs:
                """Simple ForwardFn."""
                check(
                    input_batch=inputs["input_batch"],
                    expected={
                        "input_ids": PartitionSpec(("data", "seq")),
                        "target_labels": PartitionSpec(("data", "seq")),
                        "target_num_bytes": PartitionSpec(("data", "seq")),
                    },
                )
                loss = -jax.nn.log_softmax(model_params["w"] + model_params["b"])[1]
                output_collection = new_output_collection()
                output_collection.state_updates["w"] = model_params["w"] + 1
                output_collection.state_updates["loss"] = WeightedScalar(loss, 1)
                return ForwardOutputs(loss=loss, aux={}, output_collection=output_collection)

            forward_key, param_noise_key = jax.random.split(jax.random.PRNGKey(0), 2)
            inputs = dict(
                input_batch=input_batch,
                forward_key=forward_key,
                param_noise_key=param_noise_key,
            )

            # Compute grads and loss with the minibatch decorator.
            
            loss_fn = gradient_accumulation.with_minibatch_steps(
                steps=steps, metric_accumulator=MetricAccumulator.default_config(),
                minibatch_partitioner=config_for_function(partition_by_path_rank).set(
                path_rank_to_partition={
                    # Shard batch dim on all available axis
                    (None, 1): PartitionSpec(("data", "expert", "fsdp", "seq")),
                    (None, 2): PartitionSpec(("data", "expert", "fsdp", "seq"), None),
                }
            ),
            )(loss_fn)
            
            pjit(loss_fn, in_shardings=None).lower(model_params=params, inputs=inputs).compile()


class TestMinibatchSteps(test_utils.TestCase):
    """Test `with_minibatch_steps` decorator."""

    @parameterized.named_parameters(
        ("one_step", 1),  # no accumulation
        ("two_steps", 2),
        ("four_steps", 4),
    )
    def test_minibatch_steps_grads_and_loss(self, steps):
        """Tests grad accumulation with minibatch steps."""
        params = dict(
            w=jnp.asarray([0.0, 2.0, 2.0, -3.0]),
            b=jnp.asarray([0.0, -1.0, 0.0, 0.0]),
        )

        def loss_fn(*, model_params, inputs) -> ForwardOutputs:
            """Simple ForwardFn."""
            del inputs
            loss = -jax.nn.log_softmax(model_params["w"] + model_params["b"])[1]
            output_collection = new_output_collection()
            output_collection.state_updates["w"] = model_params["w"] + 1
            output_collection.state_updates["loss"] = WeightedScalar(loss, 1)
            return ForwardOutputs(loss=loss, aux={}, output_collection=output_collection)

        batch_key, forward_key, param_noise_key = jax.random.split(jax.random.PRNGKey(0), 3)
        inputs = dict(
            input_batch=jax.random.randint(batch_key, (32, 4096), 1, 100),
            forward_key=forward_key,
            param_noise_key=param_noise_key,
        )
        # Compute grads and loss without the minibatch decorator.
        loss_expected, grads_expected = jax.value_and_grad(
            lambda x: loss_fn(model_params=x, inputs=inputs).loss
        )(params)
        # Compute grads and loss with the minibatch decorator.
        loss_fn = gradient_accumulation.with_minibatch_steps(
            steps=steps, metric_accumulator=MetricAccumulator.default_config()
        )(loss_fn)
        loss_minibatch, grads_minibatch = jax.value_and_grad(
            lambda x: loss_fn(model_params=x, inputs=inputs).loss
        )(params)

        chex.assert_trees_all_close(
            (loss_expected, grads_expected),
            (loss_minibatch, grads_minibatch),
        )


if __name__ == "__main__":
    absltest.main()
