import jax
import numpy as np
import seqio
import tensorflow as tf
import torch
from transformers import AutoConfig, LlamaForCausalLM

from axlearn.common import config, evaler, input_tf_data, measurement, utils
from axlearn.common.config import config_for_function
from axlearn.common.decoder import LmHead
from axlearn.common.decoding import StopOnSubsequence
from axlearn.common.inference import InferenceRunner
from axlearn.common.inference_pipeline import pop_string_tensors
from axlearn.common.input_lm import lm_text_preprocessor, text2text_lm_input, text_to_lm_eval_input
from axlearn.common.module import functional
from axlearn.experiments import get_named_trainer_config
from axlearn.experiments.text.common import vocab
from axlearn.experiments.text.gpt import c4_trainer
from axlearn.vision import image_classification, input_image, resnet
from utils import parameters_from_llama, parameters_to_llama

seed = 123
# config_name = "fuji-1B-v3"
config_name = "fuji-7B-v2"

# checkpoint_path = "/fsx/thiamha/fs/runs/artifacts/fs_main2/tg6.7b/241122003124/axlearn_out/checkpoints/step_00004000"
if config_name == "fuji-1B-v3":
    checkpoint_path = "/fsx/czhenguo/Projects/fruitstand/runs/artifacts/axlearn_venv/test_01/11132/axlearn_out/checkpoints/step_00000006"
    # checkpoint_path = "/fsx/czhenguo/Projects/fruitstand/runs/artifacts/axlearn_venv/test_01/11130/axlearn_out/checkpoints/step_00015000"
    sentencepiece_model_name = "bpe_128k_c4.model"
else:
    checkpoint_path = "/fsx/czhenguo/Projects/fruitstand/runs/artifacts/axlearn_venv/baselines/10976/axlearn_out/checkpoints/step_00034000"
    sentencepiece_model_name = "bpe_32k_c4.model"
# fuji-1B-v3
# checkpoint_path = "/fsx/czhenguo/Projects/fruitstand/runs/artifacts/axlearn_venv/trn_baselines/611/axlearn_out/checkpoints/step_00010000"
trainer_config_fn = get_named_trainer_config(
    config_name, config_module="axlearn.experiments.text.gpt.c4_trainer"
)
trainer_config = trainer_config_fn()
use_transformers = True


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


def init_infer_runner():
    # trainer_config.set(dir=checkpoint_path)

    devices = utils.create_device_mesh(mesh_shape=trainer_config.mesh_shape)
    mesh = jax.sharding.Mesh(devices, trainer_config.mesh_axis_names)
    infer_runner_config = InferenceRunner.config_from_trainer(trainer_config)
    infer_runner_config.init_state_builder.set(dir=checkpoint_path)
    infer_runner = infer_runner_config.instantiate(parent=None)
    return infer_runner, infer_runner_config, mesh


# infer_runner, infer_runner_config, mesh = init_infer_runner()


def run_inference(texts):
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
        prng_key = jax.random.PRNGKey(seed=1)
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
            # sentence_piece_vocab.tokenizer.bos_id()  # -1

            results.extend(output_texts)
            print(output_texts)
    return results


def validate_conversion(fuji_model_name, llama_model_name, load_true_model=False, reverse=False):
    """Validate conversion between fuji and llama model."""
    trainer_config_map = c4_trainer.named_trainer_configs()
    trainer_config_fn = trainer_config_map[fuji_model_name]
    trainer_config = trainer_config_fn()
    model_config = trainer_config.model
    model_config.set(name="fuji-test-model")

    if fuji_model_name == "fuji-7B-v2":
        # llama2 7B does not share lm_head with embedding, but fuji does
        # need to disable lm_head sharing for fuji to match llama
        # model_config.decoder.set(lm_head=None)
        model_config.decoder.set(lm_head=LmHead.default_config())

    # initialize transformer model
    if load_true_model:
        # load model to a different device to avoid OOM
        llama = LlamaForCausalLM.from_pretrained(llama_model_name, local_files_only=True)
    else:
        # self-specify smaller config for easier validation
        config = AutoConfig.from_pretrained(
            f"{llama_model_name}_config.json",
            local_files_only=True,
        )
        llama = LlamaForCausalLM._from_config(config)

        # adjust num_layers to match the value in {llama_model_name}_config.json
        model_config.decoder.transformer.set(num_layers=2)

    # fuji model has different vocab size even for the same model size
    # this only allows you to convert true llama model to fuji, the reverse is not valid for true model
    model_config.decoder.set(vocab_size=llama.config.vocab_size)

    # llama.to("cuda:2")
    llama = llama.eval()

    # initialize fuji model
    fuji = model_config.instantiate(parent=None)
    prng_key = jax.random.PRNGKey(0)
    state = fuji.initialize_parameters_recursively(prng_key=prng_key)

    # generate dummy input data
    ids = jax.random.randint(jax.random.PRNGKey(123), shape=(2, 2), minval=0, maxval=12345)
    torch_ids = torch.from_numpy(np.asarray(ids))

    # conversion for llama2 and llama3 would be different
    # for example llama3 would use GQA and some of the model also share weights between lm_head and emb
    if llama_model_name in ["Llama-2-7b", "Llama-2-7b-hf"]:
        version = 2
    else:
        version = 3

    # convert params
    if reverse:
        llama_state_dict = parameters_to_llama(state, llama, version)
        llama.load_state_dict(llama_state_dict)
    else:
        state = parameters_from_llama(llama, state, version)

    input_batch = {"input_ids": ids}
    (_, aux), _ = functional(
        fuji,
        is_training=False,
        prng_key=jax.random.PRNGKey(123),
        state=state,
        inputs={"input_batch": input_batch, "return_aux": True},
    )

    with torch.no_grad():
        output = llama(torch_ids)

    fuji_logits = np.asarray(aux["logits"])
    llama_logits = output.logits.numpy()

    # The difference is caused by the SDPA attention layer. The deeper the larger the error.
    if fuji_model_name == "fuji-1B-v3":
        atol = 3e-2
    elif fuji_model_name == "fuji-3B-v3":
        atol = 2e-2
    elif fuji_model_name == "fuji-7B-v2":
        atol = 3e-2
    elif fuji_model_name == "fuji-8B-v3":
        atol = 2e-1
    elif fuji_model_name == "fuji-70B-v3":
        atol = 2.0
    else:
        atol = 2e-3
    assert np.allclose(fuji_logits, llama_logits, atol=atol), (
        f"{fuji_logits[0,0,:10]} != {llama_logits[0,0,:10]}, "
        f"{np.abs(fuji_logits - llama_logits).max()}"
    )


def load_checkpoint(trainer_config, model):
    if use_transformers:
        pass
    else:
        checkpointer_config = trainer_config.cehckpointer
        checkpointer_config.dir = checkpoint_path
        checkpointer = checkpointer_config.instantiate(parent=None)
        state = model.initialize_parameters_recursively(prng_key=prng_key)
        step, state = checkpointer.restore(state=state)

    return state


def get_fuji_with_llama_weights():
    model_config = trainer_config.model
    model_config.set(name="fuji-test-model")

    if config_name == "fuji-7B-v2":
        # llama2 7B does not share lm_head with embedding, but fuji does
        # need to disable lm_head sharing for fuji to match llama
        model_config.decoder.set(lm_head=LmHead.default_config())

    llama = LlamaForCausalLM.from_pretrained("Llama-2-7b-hf", local_files_only=True)
    prng_key = jax.random.PRNGKey(0)
    model = model_config.instantiate(parent=None)
    model_state = model.initialize_parameters_recursively(prng_key=prng_key)
    model_state = parameters_from_llama(llama, model_state, 2)
    return model, model_state


def generate(texts):
    # TODO init model and load checkpoint without InferenceRunner
    # model = infer_runner.model
    # model_state = infer_runner._inference_runner_state.model
    # model_config = trainer_config.model
    # model_state = load_checkpoint()
    model, model_state = get_fuji_with_llama_weights()

    # init tokenizer for decode
    if use_transformers:
        # update pad_token_id since they are different in fuji and llama
        tokenizer = get_transformers_tokenizer()
        pad_token_id = model.config.decoder.pad_token_id
        tokenizer.pad_token_id = pad_token_id

        # Fuji models use different eos_token_id than llama models
        # fuji eos_token_id = 0, llama eos_token_id = 128001 (llama3.2 1B)
        stop_decoding_condition = StopOnSubsequence([[tokenizer.eos_token_id]])
    else:
        vocab_cfg = config_for_function(vocab).set(
            sentencepiece_model_name=sentencepiece_model_name, num_extra_ids=None
        )
        tokenizer = vocab_cfg.instantiate()
        stop_decoding_condition = StopOnSubsequence([[model.decoder.config.eos_token_id]])

    results = list()
    batch_size, max_len = 8, 4096
    method = "sample_decode"
    # method="beam_search_decode"
    input_ids = tokenizer.batch_encode_plus(texts, padding="max_length", max_length=max_len)[
        "input_ids"
    ]

    # TODO decoder batch mode
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

        # TODO how to get model states with invocation context without using infer_runner? https://github.com/apple/axlearn/blob/a15a3bcbb976c14db157a8958df368a48c614c1f/axlearn/experiments/text/gpt/param_converter_test.py#L105
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
        batch, num, indices = jax.numpy.where(output.sequences == tokenizer.eos_token_id)
        if indices.size > 0:
            output_texts = tokenizer.batch_decode([output.sequences[0][0][: indices[0]]])
        else:
            # in case eos_token is not generated
            output_texts = tokenizer.batch_decode(output.sequences[0])
        print(output_texts)
        results.extend(output_texts)
    return results


if __name__ == "__main__":
    texts = [
        "How are you doing?",
        "who is the president of the US now?",
        "The USA is in which continent?",
        "California is a state in",
        "Can you tell me something about California state?\n",
        "California is a state in",
        "California is a state in",
        "California is a state in",
    ]
    # run_inference(texts)
    # generate(texts)
    # validate_conversion("fuji-1B-v3", "Llama-3.2-1B", load_true_model=True)
    # validate_conversion("fuji-7B-v2", "Llama-2-7b-hf", load_true_model=True)
    # validate_conversion("fuji-7B-v2", "Llama-2-7b-hf", load_true_model=False)
    validate_conversion("fuji-7B-v2", "Llama-2-7b-hf", load_true_model=False, reverse=True)
