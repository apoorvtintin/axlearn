from collections import defaultdict

import jax
import numpy as np
import seqio
import tensorflow as tf
import torch
from jax import numpy as jnp
from transformers import AutoConfig, LlamaForCausalLM

from axlearn.common import config, evaler, input_tf_data, measurement, utils
from axlearn.common.config import config_for_function
from axlearn.common.decoder import LmHead
from axlearn.common.decoding import StopOnSubsequence
from axlearn.common.inference_pipeline import pop_string_tensors
from axlearn.common.input_lm import lm_text_preprocessor, text2text_lm_input, text_to_lm_eval_input
from axlearn.common.module import functional
from axlearn.experiments.text.common import vocab
from axlearn.experiments.text.gpt import c4_trainer
from utils import (
    backend,
    copy_files,
    get_fuji_and_llama,
    get_mesh,
    get_trainer_config,
    init_infer_runner,
    load_checkpoint,
    parameters_from_llama,
    parameters_to_llama,
    save_axlearn_checkpoint,
    save_transformers_checkpoint,
    seed,
)

sentencepiece_model_name = "bpe_32k_c4.model"
use_transformers = False


def get_transformers_tokenizer():
    """Replace sentence piece tokenizer with transformers tokenizer when loading Llama checkpoints."""
    from transformers import AutoTokenizer

    if config_name == "fuji-1B-v3":
        tokenizer = AutoTokenizer.from_pretrained("Llama-3.2-1B")
    else:
        tokenizer = AutoTokenizer.from_pretrained("Llama-2-7b-hf")
    return tokenizer


def make_ds_fn(
    is_training: bool, texts: list[str], repeat: int = 1
) -> input_tf_data.BuildDatasetFn:
    del is_training

    def ds_fn() -> tf.data.Dataset:
        def data_gen():
            for _ in range(repeat):
                for index, text in enumerate(texts):
                    yield {"text": text, "index": index}

        return tf.data.Dataset.from_generator(
            data_gen,
            output_signature={
                "text": tf.TensorSpec(shape=(), dtype=tf.string),
                "index": tf.TensorSpec(shape=(), dtype=tf.uint32),
            },
        )

    return ds_fn


def run_inference(texts, config_name, checkpoint_path):
    trainer_config = get_trainer_config(config_name)
    infer_runner, infer_runner_config = init_infer_runner(trainer_config, checkpoint_path)
    mesh = get_mesh(trainer_config)
    model = infer_runner.model
    evaler_config = trainer_config.evalers["validation"]
    evaler_config.name = "validation"
    evaler_config.summary_writer.dir = "/fsx/czhenguo/Projects/fruitstand/runs/artifacts/axlearn_venv/ttest_01/11126/axlearn_out/summaries/validation"

    # init tokenizer for decode
    vocab_cfg = config_for_function(vocab).set(
        sentencepiece_model_name=sentencepiece_model_name, num_extra_ids=None
    )
    sentence_piece_vocab = vocab_cfg.instantiate()

    batch_size, max_len = 8, 4096
    evaler_config.input = evaler_config.input.set(
        source=config_for_function(make_ds_fn).set(texts=texts),
        processor=config_for_function(text_to_lm_eval_input).set(
            vocab_cfg=vocab_cfg,
            max_len=max_len,
            replace_newlines_with="\n",
            stride=2,
        ),
        batcher=evaler_config.input.batcher.set(global_batch_size=batch_size),
    )

    results = list()

    with mesh:
        model_param_specs = model.create_parameter_specs_recursively()
        model_param_partition_specs = jax.tree.map(lambda spec: spec.mesh_axes, model_param_specs)
        evaler = evaler_config.instantiate(
            parent=None,
            model=model,
            model_param_partition_specs=model_param_partition_specs,
        )
        eval_input_iter = iter(evaler.input.dataset())
        prng_key = jax.random.PRNGKey(seed=seed)
        method_runner = infer_runner.create_method_runner(method="predict", prng_key=prng_key)

        for batch_ix, input_batch in enumerate(evaler.input.batches(eval_input_iter)):
            input_ids = input_batch["input_ids"].tolist()
            input_texts = sentence_piece_vocab.tokenizer.decode_ids(input_ids)
            input_batch, input_batch_str_tensors = pop_string_tensors(input_batch)
            input_batch = utils.as_numpy_array(input_batch)
            global_input_batch = utils.host_to_global_device_array(
                input_batch, partition=infer_runner_config.input_batch_partition_spec
            )
            output = method_runner(global_input_batch)
            output_batch = utils.global_to_host_array(
                output.output_batch,
                partition=infer_runner_config.input_batch_partition_spec,
            )
            output_batch = utils.global_to_host_array(
                output.output_batch,
                partition=infer_runner_config.input_batch_partition_spec,
            )
            # (16, 4096, 32768)
            logits = output.output_batch["logits"]
            output_ids = jax.numpy.argmax(logits, axis=-1)
            output_texts = sentence_piece_vocab.tokenizer.decode_ids(output_ids.tolist())
            # sentence_piece_vocab.tokenizer.pad_id()  # 0
            # sentence_piece_vocab.tokenizer.eos_id()  # 1
            # sentence_piece_vocab.tokenizer.unk_id()  # 2
            # sentence_piece_vocab.tokenizer.bos_id()  # -1
            # sentence_piece_vocab.tokenizer.id_to_piece(0) <pad>
            # sentence_piece_vocab.tokenizer.id_to_piece(1) </s>
            # sentence_piece_vocab.tokenizer.id_to_piece(2) <unk>

            results.extend(output_texts)
            print(output_texts)
    return results


def get_sentence_piece_tokenizer():
    vocab_cfg = config_for_function(vocab).set(
        sentencepiece_model_name=sentencepiece_model_name, num_extra_ids=None
    )
    sentence_piece_vocab = vocab_cfg.instantiate()
    return sentence_piece_vocab


def validate_conversion(
    fuji_model_name,
    llama_model_name,
    load_true_model=False,
    reverse=False,
    texts=None,
    fuji_model_path=None,
    use_gqa=False,
    trn_checkpoint=False,
):
    """Run a forward pass and compare logits to validate conversion between fuji and llama model."""
    fuji, state, llama = get_fuji_and_llama(
        fuji_model_name, llama_model_name, load_true_model, reverse, fuji_model_path=fuji_model_path
    )

    tokenizer = get_sentence_piece_tokenizer()

    def pad_list(l, max_len, fill_value=tokenizer.pad_id):
        return [tokenizer.eos_id] + l + [fill_value] * (max_len - len(l) - 1)

    if texts:
        # end to end test with texts
        ids = tokenizer.encode(texts)
    else:
        # generate dummy input data
        ids = jax.random.randint(jax.random.PRNGKey(seed), shape=(2, 2), minval=0, maxval=12345)

    padded_ids = [pad_list(cur_ids, 6) for cur_ids in ids]
    ids = [cur_ids[:4] for cur_ids in padded_ids]
    target_ids = [cur_ids[1:5] for cur_ids in padded_ids]
    torch_ids = torch.from_numpy(np.asarray(ids))
    torch_target_ids = torch.from_numpy(np.asarray(target_ids))

    # convert params
    if reverse:
        llama_state_dict = parameters_to_llama(
            state, llama, use_gqa=use_gqa, trn_checkpoint=trn_checkpoint
        )
        llama.load_state_dict(llama_state_dict)
    else:
        state = parameters_from_llama(llama, state, use_gqa=use_gqa, trn_checkpoint=trn_checkpoint)

    input_batch = {"input_ids": jnp.asarray(ids), "target_labels": jnp.asarray(target_ids)}
    (loss, aux), output_collection = functional(
        fuji,
        is_training=False,
        prng_key=jax.random.PRNGKey(seed),
        state=state,
        inputs={"input_batch": input_batch, "return_aux": True},
    )

    with torch.no_grad():
        output = llama(torch_ids, labels=torch_target_ids)
        # transformers will shift the ids when calculating the loss
        # so to make sure the loss would match, adjust the input
        # https://github.com/huggingface/transformers/blob/641adca55832ed9c5648f54dcd8926d67d3511db/src/transformers/models/llama/modeling_llama.py#L833
        extra_ids = [cur_ids[:5] for cur_ids in padded_ids]
        extra_ids = torch.from_numpy(np.asarray(extra_ids))
        output_with_loss = llama(extra_ids, labels=extra_ids)
        llama_loss = output_with_loss.loss

    fuji_logits = np.asarray(aux["logits"])
    llama_logits = output.logits.numpy()
    assert isinstance(aux["logits"].dtype, np.dtypes.Float32DType)

    # TODO should not do the softmax myself
    fuji_probs = np.asarray(jax.nn.softmax(aux["logits"]))
    llama_probs = torch.softmax(output.logits, dim=-1).numpy()
    assert isinstance(fuji_probs.dtype, np.dtypes.Float32DType)

    np.save(f"{fuji_model_name}_probs", fuji_probs)
    np.save(f"{llama_model_name}_probs", llama_probs)


def run_forward_pass(
    fuji_model_name,
    texts,
    fuji_model_path,
):
    """Run forward pass on text on TRN device.

    changes from this commit are needed to get logit output. Also, this only works in TRN device,
    because TRN branch has changes on functions like utils.host_to_global_device_array
    """
    trainer_config = get_trainer_config(fuji_model_name)
    infer_runner, infer_runner_config = init_infer_runner(trainer_config, fuji_model_path)
    fuji = infer_runner.model
    state = infer_runner._inference_runner_state.model

    tokenizer = get_sentence_piece_tokenizer()

    def pad_list(l, max_len, fill_value=tokenizer.pad_id):
        return [tokenizer.eos_id] + l + [fill_value] * (max_len - len(l) - 1)

    ids = tokenizer.encode(texts)
    padded_ids = [pad_list(cur_ids, 6) for cur_ids in ids]
    ids = [cur_ids[:4] for cur_ids in padded_ids]
    target_ids = [cur_ids[1:5] for cur_ids in padded_ids]

    trainer_config = get_trainer_config(fuji_model_name)
    trainer_config.dir = "runs/artifacts/validate_trn/axlearn_out/"
    trainer_config.evalers["validation"] = trainer_config.evalers["validation"].set(eval_dtype=jnp.float32)
    trainer_config.evalers["validation"].metric_calculator = trainer_config.evalers[
        "validation"
    ].metric_calculator.set(model_method_kwargs={"return_aux": True})
    evaler_config = trainer_config.evalers["validation"]
    evaler_config.name = "validation"
    evaler_config.summary_writer.dir = (
        "runs/artifacts/validate_trn/axlearn_out/summaries/validation"
    )
    evaler_config.set(eval_dtype=jnp.float32)

    with get_mesh(trainer_config):
        model_param_specs = fuji.create_parameter_specs_recursively()
        model_param_partition_specs = jax.tree.map(lambda spec: spec.mesh_axes, model_param_specs)
        evaler = evaler_config.instantiate(
            parent=None,
            model=fuji,
            model_param_partition_specs=model_param_partition_specs,
        )

        prng_key = jax.random.PRNGKey(seed)
        input_batch = {"input_ids": jnp.asarray(ids), "target_labels": jnp.asarray(target_ids)}
        if backend == "neuron":
            global_input_batch = utils.host_to_global_device_array(
                input_batch,
                partition=trainer_config.input_partition_type,
                batch_axis_names=trainer_config.batch_axis_names,
            )
        else:
            global_input_batch = utils.host_to_global_device_array(
                input_batch,
                partition=infer_runner_config.input_batch_partition_spec
            )
        # model param not actually used in ModelSummaryAccumulator
        metric_calculator_state = evaler.metric_calculator.init_state(
            prng_key=prng_key, model_params=state
        )
        next_key, forward_prng = jax.random.split(prng_key)
        forward_outputs = evaler.metric_calculator.forward(
            global_input_batch,
            model_params=state,
            state=metric_calculator_state,
        )

        loss, fuji_outputs = forward_outputs["output"]
        fuji_logits = np.asarray(fuji_outputs["logits"])
        fuji_probs = np.asarray(jax.nn.softmax(fuji_logits))

    np.save(f"{fuji_model_name}_probs", fuji_probs)
    assert isinstance(fuji_logits.dtype, np.dtypes.Float32DType)
    assert isinstance(fuji_probs.dtype, np.dtypes.Float32DType)
    return fuji_probs


def convert_and_save_checkpoint(
    fuji_model_name,
    llama_model_name,
    load_true_model=True,
    reverse=False,
    fuji_model_path=None,
    save_name="converted_model",
    use_gqa=False,
    trn_checkpoint=False,
):
    fuji, state, llama = get_fuji_and_llama(
        fuji_model_name,
        llama_model_name,
        load_true_model=load_true_model,
        reverse=reverse,
        fuji_model_path=fuji_model_path,
    )

    # convert params
    if reverse:
        llama_state_dict = parameters_to_llama(
            state, llama, use_gqa=use_gqa, trn_checkpoint=trn_checkpoint
        )
        llama.load_state_dict(llama_state_dict)
        checkpoint_path = (
            f"/fsx/czhenguo/Projects/fruitstand/runs/artifacts/axlearn_to_transformers/{save_name}"
        )
        save_transformers_checkpoint(llama, checkpoint_path)
        copy_files(llama_model_name, checkpoint_path)
    else:
        state = parameters_from_llama(llama, state, use_gqa=use_gqa, trn_checkpoint=trn_checkpoint)
        checkpoint_path = (
            f"/fsx/czhenguo/Projects/fruitstand/runs/artifacts/transformers_to_axlearn/{save_name}"
        )

        trainer_config = get_trainer_config(fuji_model_name)
        save_axlearn_checkpoint(fuji, state, checkpoint_path, get_mesh(trainer_config))

    print(f"Successfuly save checkpoint to {checkpoint_path}.")


def get_fuji_with_llama_weights(config_name):
    trainer_config = get_trainer_config(config_name)
    model_config = trainer_config.model
    model_config.set(name="fuji-test-model")

    if config_name == "fuji-7B-v2":
        # llama2 7B does not share lm_head with embedding, but fuji does
        # need to disable lm_head sharing for fuji to match llama
        model_config.decoder.set(lm_head=LmHead.default_config())

    llama = LlamaForCausalLM.from_pretrained("Llama-2-7b-hf", local_files_only=True)
    prng_key = jax.random.PRNGKey(seed)
    model = model_config.instantiate(parent=None)
    model_state = model.initialize_parameters_recursively(prng_key=prng_key)
    model_state = parameters_from_llama(llama, model_state, 2)
    return model, model_state


def generate(texts, config_name, checkpoint_path):
    results = list()
    batch_size, max_len = 8, 4096
    trainer_config = get_trainer_config(config_name)

    # init tokenizer for decode
    if use_transformers:
        # model, model_state = get_fuji_with_llama_weights(config_name)
        model_config = trainer_config.model
        model_config.set(name="model")

        model = model_config.instantiate(parent=None)
        model_state = load_checkpoint(trainer_config, checkpoint_path)

        # update pad_token_id since they are different in fuji and llama
        tokenizer = get_transformers_tokenizer()
        pad_token_id = model.config.decoder.pad_token_id
        tokenizer.pad_token_id = pad_token_id
        eos_token_id = tokenizer.eos_token_id

        # Fuji models use different eos_token_id than llama models
        # fuji eos_token_id = 0, llama eos_token_id = 128001 (llama3.2 1B)
        stop_decoding_condition = StopOnSubsequence([[tokenizer.eos_token_id]])

        input_ids = tokenizer.batch_encode_plus(texts, padding="max_length", max_length=max_len)[
            "input_ids"
        ]
    else:
        model_config = trainer_config.model
        model_config.set(name="model")

        model = model_config.instantiate(parent=None)
        model_state = load_checkpoint(trainer_config, checkpoint_path)

        vocab_cfg = config_for_function(vocab).set(
            sentencepiece_model_name=sentencepiece_model_name, num_extra_ids=None
        )
        tokenizer = vocab_cfg.instantiate()
        eos_token_id = tokenizer.eos_id
        stop_decoding_condition = StopOnSubsequence([[model.decoder.config.eos_token_id]])

        input_ids = tokenizer.encode(texts)

        # add padding
        def pad_list(l, max_len, fill_value=tokenizer.pad_id):
            # need to add a eos_token_id at the begining to be consistent with text_to_lm_eval_input
            return [eos_token_id] + l + [fill_value] * (max_len - len(l) - 1)

        input_ids = [pad_list(ids, max_len) for ids in input_ids]
        # input_ids = np.pad(input_ids, ((0, 0),(0, max_len)), "constant", constant_values=0)

    method = "sample_decode"
    # method="beam_search_decode"

    for input_id in input_ids:
        # follow decoder input format https://github.com/apple/axlearn/blob/a15a3bcbb976c14db157a8958df368a48c614c1f/axlearn/common/decoder_test.py#L569
        input_batch = {
            "input_batch": {"prefix": jax.numpy.asarray([input_id])},
        }
        # Override the default decoder eos_token_id since fuji and llama has different eos_token_id
        # but beam_search_decode does not accept this argument
        # https://github.com/apple/axlearn/blob/a15a3bcbb976c14db157a8958df368a48c614c1f/axlearn/common/decoder.py#L336
        if method == "sample_decode":
            input_batch["stop_decoding_condition"] = stop_decoding_condition
        # TODO add mask for batch running https://github.com/apple/axlearn/blob/a15a3bcbb976c14db157a8958df368a48c614c1f/axlearn/common/decoder_test.py#L563C13-L563C24

        # TODO the decoding process does not seem to start from the last token, but start from the first token when testing with axlearn model
        output, _ = functional(
            model,
            is_training=False,
            prng_key=jax.random.PRNGKey(seed=seed),
            state=model_state,
            inputs=input_batch,
            method=method,
        )
        # need to manually remove tokens after eos_token if using sample_decode
        # because it will call _decode_init and create a sequence with the length max_len
        # https://github.com/apple/axlearn/blob/main/axlearn/common/decoding.py#L790
        batch, num, indices = jax.numpy.where(output.sequences == eos_token_id)
        if use_transformers:
            decode_fn = tokenizer.batch_decode
        else:
            decode_fn = lambda x: tokenizer.decode(x[0])

        if indices.size > 0:
            output_texts = decode_fn([output.sequences[0][0][: indices[0]]])
        else:
            # in case eos_token is not generated
            output_texts = decode_fn(output.sequences[0][0])
        print(output_texts)
        results.extend([output_texts])
    return results


def extend_texts(texts, batch_size=16):
    results = list()
    while len(results) < batch_size:
        results.extend(texts)

    return results[:batch_size]


if __name__ == "__main__":
    texts = [
        "How are you doing?",
        "who is the president of the US now?",
        "The USA is in which continent?",
        "California is a state in",
        "Can you tell me something about California state?\n",
    ]
    # config_name = "fuji-7B-v2"
    # checkpoint_path = "/fsx/czhenguo/Projects/fruitstand/runs/artifacts/axlearn_venv/baselines/10976/axlearn_out/checkpoints/step_00034000"
    # run_inference(texts, config_name, checkpoint_path)
    # generate(texts, config_name, checkpoint_path)

    # Llama to Axlearn 7B GPU true model
    # validate_conversion(
    #     "fuji-7B-v2",
    #     "Llama-2-7b-hf",
    #     load_true_model=True,
    #     texts=texts,
    #     trn_checkpoint=False,
    #     use_gqa=False,
    # )

    # Llama to Axlearn 7B GPU dummy model
    # validate_conversion(
    #     "fuji-7B-v2",
    #     "Llama-2-7b-hf",
    #     load_true_model=False,
    #     texts=texts,
    #     trn_checkpoint=False,
    #     use_gqa=False,
    # )

    # Axlearn to Llama 7B GPU true model
    # validate_conversion(
    #     "fuji-7B-v2",
    #     "Llama-2-7b-hf",
    #     load_true_model=True,
    #     reverse=True,
    #     texts=texts,
    #     fuji_model_path="/fsx/czhenguo/Projects/fruitstand/runs/artifacts/axlearn_venv/baselines/10976/axlearn_out/checkpoints/step_00034000",
    #     trn_checkpoint=False,
    #     use_gqa=False,
    # )

    # texts = extend_texts(texts, 16)
    # run_forward_pass(
    #     "fuji-7B-v2",
    #     texts,
    #     "/fsx/czhenguo/Projects/fruitstand/runs/artifacts/axlearn_venv/validation/fuji-7B-v2-4l/step_00022794",
    # )

    # Axlearn to Llama 7B TRN 4L true model
    validate_conversion(
        "fuji-7B-v2",
        "Llama-2-7b-hf",
        load_true_model=True,
        reverse=True,
        texts=texts,
        # fuji_model_path="/fsx/czhenguo/Projects/fruitstand/runs/artifacts/axlearn_venv/baselines/10976/axlearn_out/checkpoints/step_00034000",
        fuji_model_path="/fsx/czhenguo/Projects/fruitstand/runs/artifacts/axlearn_venv/validation/fuji-7B-v2-4l/step_00022794",
        trn_checkpoint=True,
        use_gqa=False,
    )

    # Axlearn to Llama 70B GPU true model
    # validate_conversion(
    #     "fuji-70B-v2",
    #     "Llama-2-70b-hf",
    #     load_true_model=True,
    #     reverse=True,
    #     texts=texts,
    #     fuji_model_path="/fsx/czhenguo/Projects/fruitstand/runs/artifacts/axlearn_venv/baselines/10985/axlearn_out/checkpoints/step_00035000",
    #     trn_checkpoint=False,
    #     use_gqa=True,
    # )

    # Llama to Axlearn 7B TRN dummy model
    # validate_conversion(
    #     "fuji-7B-v2",
    #     "Llama-2-7b-hf",
    #     load_true_model=False,
    #     texts=texts,
    #     trn_checkpoint=True,
    #     use_gqa=False,
    # )

    # Llama to Axlearn 7B TRN dummy model
    # validate_conversion(
    #     "fuji-70B-v2",
    #     "Llama-2-70b-hf",
    #     load_true_model=False,
    #     texts=texts,
    #     trn_checkpoint=True,
    #     use_gqa=True,
    # )

    # Axlearn to Llama 70B TRN true model
    # validate_conversion(
    #     "fuji-70B-v2",
    #     "Llama-2-70b-hf",
    #     load_true_model=True,
    #     reverse=True,
    #     texts=texts,
    #     fuji_model_path="/fsx/czhenguo/Projects/fruitstand/runs/artifacts/241230232345/axlearn_out/checkpoints/step_00000002",
    #     trn_checkpoint=True,
    #     use_gqa=True,
    # )

    # convert_and_save_checkpoint(
    #     "fuji-7B-v2",
    #     "Llama-2-7b-hf",
    #     load_true_model=True,
    #     reverse=False,
    #     save_name="Llama-2-7b-trn",
    #     trn_checkpoint=True,
    #     use_gqa=False,
    # )
    # convert_and_save_checkpoint(
    #     "fuji-7B-v2",
    #     "Llama-2-7b-hf",
    #     load_true_model=True,
    #     reverse=True,
    #     save_name="Llama-2-7b-hf",
    #     fuji_model_path="/fsx/czhenguo/Projects/fruitstand/runs/artifacts/transformers_to_axlearn/Llama-2-7b-trn/step_00022794",
    #     trn_checkpoint=True,
    #     use_gqa=False,
    # )
    # convert_and_save_checkpoint(
    #     "fuji-7B-v2",
    #     "Llama-2-7b-hf",
    #     load_true_model=True,
    #     reverse=True,
    #     fuji_model_path="/fsx/czhenguo/Projects/fruitstand/runs/artifacts/axlearn_venv/baselines/10976/axlearn_out/checkpoints/step_00034000",
    #     save_name="baseline_34000",
    #     trn_checkpoint=False,
    #     use_gqa=False,
    # )
    # convert_and_save_checkpoint(
    #     "fuji-7B-v2",
    #     "/fsx/czhenguo/Projects/fruitstand/runs/artifacts/axlearn_to_transformers/baseline_34000",
    #     load_true_model=True,
    #     reverse=False,
    #     save_name="round_trip",
    #     trn_checkpoint=False,
    #     use_gqa=False,
    # )
    # convert_and_save_checkpoint(
    #     "fuji-7B-v2",
    #     "Llama-2-7b-hf",
    #     load_true_model=False,
    #     reverse=True,
    #     save_name="random_init",
    #     trn_checkpoint=False,
    #     use_gqa=False,
    # )
    # convert_and_save_checkpoint(
    #     "fuji-70B-v2",
    #     "Llama-2-70b-hf",
    #     load_true_model=True,
    #     reverse=True,
    #     fuji_model_path="/fsx/czhenguo/Projects/fruitstand/runs/artifacts/241230232345/axlearn_out/checkpoints/step_00000002",
    #     save_name="trn_70B",
    #     trn_checkpoint=True,
    #     use_gqa=True,
    # )
    # convert_and_save_checkpoint(
    #     "fuji-70B-v2",
    #     "Llama-2-70b-hf",
    #     load_true_model=True,
    #     reverse=False,
    #     save_name="Llama-2-70b-hf-gpu",
    #     trn_checkpoint=False,
    #     use_gqa=True,
    # )
