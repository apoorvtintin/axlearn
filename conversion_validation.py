import numpy as np
import pandas as pd

DEFAULT_K = 64
DEFAULT_P = 0.95


def relative_difference(x, y):
    # May get hugh relative difference since some logits would be super small
    return np.abs(x - y) / np.min([np.abs(x), np.abs(y)], axis=0)


def average_top_k_jaccard_similarity(x, y, max_k=100):
    ks = [i for i in range(1, max_k + 1, 5)]
    jaccard_indices = defaultdict(int)
    batch_size, seq_len, vocab_size = x.shape

    for x_sequence, y_sequence in zip(x, y):
        for x_logits, y_logits in zip(x_sequence, y_sequence):
            for k in ks:
                top_x_indices = x_logits.argsort()[::-1]
                top_y_indices = y_logits.argsort()[::-1]
                top_x_set = set(top_x_indices[:k])
                top_y_set = set(top_y_indices[:k])
                jaccard_index = len(top_x_set.intersection(top_y_set)) / len(
                    top_x_set.union(top_y_set)
                )
                jaccard_indices[k] += jaccard_index

    for k in jaccard_indices:
        jaccard_indices[k] /= batch_size * seq_len
    return jaccard_indices


def get_top_p(x, p):
    """Get top p from 1D array."""
    x_indices = np.argsort(x)[::-1]
    total = 0.0
    top_values = list()
    top_indices = list()

    for idx in x_indices:
        val = x[idx]
        total += val
        top_values.append(val)
        top_indices.append(idx)

        if total > p:
            break

    return top_values, top_indices


def test_top_p():
    probs = np.asarray([0.3, 0.2, 0.25, 0.15, 0.1])
    top_values, top_indices = get_top_p(probs, 0.85)
    np.testing.assert_array_equal(top_values, [0.3, 0.25, 0.2, 0.15])
    np.testing.assert_array_equal(top_indices, [0, 2, 1, 3])


def get_top_k(x, k):
    """Get top k from 1D array."""
    x_indices = np.argsort(x)[::-1]
    sorted_x = x[x_indices]

    return sorted_x[:k], x_indices[:k]


def test_top_k():
    probs = np.asarray([0.3, 0.2, 0.25, 0.15, 0.1])
    top_values, top_indices = get_top_k(probs, 4)
    np.testing.assert_array_equal(top_values, [0.3, 0.25, 0.2, 0.15])
    np.testing.assert_array_equal(top_indices, [0, 2, 1, 3])


def ndarray_apply_order(x, indices):
    # x[indices] does not work for ndarray with dimension higher than 2
    # since different indices need to be applied on different array
    result = list()
    for i, seq in enumerate(x):
        new_seq = list()
        for j, tokens in enumerate(seq):
            new_seq.append(tokens[indices[i][j]])
        result.append(new_seq)
    return np.asarray(result)


def get_top_k_union(x, y, k=DEFAULT_K):
    """Get top k in x and y 3D array, then get union among them."""
    x_indices = np.argsort(x, axis=-1)[:, :, ::-1]
    y_indices = np.argsort(y, axis=-1)[:, :, ::-1]
    x_results = list()
    y_results = list()

    for i, (x_seq, y_seq) in enumerate(zip(x, y)):
        x_seq_results = list()
        y_seq_results = list()
        for j, (x_token, y_token) in enumerate(zip(x_seq, y_seq)):
            top_x_indices = x_indices[i][j][:k]
            top_y_indices = y_indices[i][j][:k]
            top_indices = list(set(top_x_indices).union(set(top_y_indices)))
            top_x = x_token[top_indices]
            top_y = y_token[top_indices]

            x_seq_results.append(top_x)
            y_seq_results.append(top_y)
        x_results.append(x_seq_results)
        y_results.append(y_seq_results)

    return x_results, y_results


def get_top_k_smallest(x, y, k=DEFAULT_K):
    """Get top k smallest in 3D array."""
    x, y = get_top_k_union(x, y, k)
    results = list()

    for i, (x_seq, y_seq) in enumerate(zip(x, y)):
        seq_results = list()
        for j, (x_token, y_token) in enumerate(zip(x_seq, y_seq)):
            seq_results.append(np.min(np.concatenate((x_token, y_token), axis=-1)))
        results.append(seq_results)
    return results


def test_get_top_k_smallest():
    x = np.asarray([[[0.3, 0.2, 0.25, 0.15, 0.1] for _ in range(2)] for _ in range(2)])
    y = np.asarray([[[0.3, 0.19, 0.23, 0.15, 0.1] for _ in range(2)] for _ in range(2)])

    top_k_smallest = get_top_k_smallest(x, y, 3)

    np.testing.assert_array_equal(top_k_smallest, [[0.19 for _ in range(2)] for _ in range(2)])


def get_top_p_smallest(x, y, p=DEFAULT_P):
    """Get top k smallest in 3D array."""
    x, y = get_top_p_union(x, y, p)
    results = list()

    for i, (x_seq, y_seq) in enumerate(zip(x, y)):
        seq_results = list()
        for j, (x_token, y_token) in enumerate(zip(x_seq, y_seq)):
            seq_results.append(np.min(np.concatenate((x_token, y_token), axis=-1)))
        results.append(seq_results)
    return results


def test_get_top_p_smallest():
    x = np.asarray([[[0.3, 0.2, 0.25, 0.15, 0.1] for _ in range(2)] for _ in range(2)])
    y = np.asarray([[[0.3, 0.19, 0.23, 0.15, 0.1] for _ in range(2)] for _ in range(2)])

    top_p_smallest = get_top_p_smallest(x, y, 0.95)

    np.testing.assert_array_equal(top_p_smallest, [[0.1 for _ in range(2)] for _ in range(2)])


def get_p_at_top_k(x, y, k=DEFAULT_K):
    """Get p values for a given top k value."""
    x_indices = np.argsort(x, axis=-1)[:, :, ::-1]
    y_indices = np.argsort(y, axis=-1)[:, :, ::-1]

    results = list()
    for i, (x_seq, y_seq) in enumerate(zip(x, y)):
        seq_results = list()
        for j, (x_token, y_token) in enumerate(zip(x_seq, y_seq)):
            top_x_indices = x_indices[i][j][:k]
            top_y_indices = y_indices[i][j][:k]
            top_indices = list(set(top_x_indices).union(set(top_y_indices)))
            top_x = x_token[top_indices]
            top_y = y_token[top_indices]
            seq_results.append(np.mean((np.sum(top_x), np.sum(top_y))))
        results.append(seq_results)

    return results


def test_get_p_at_top_k():
    x = np.asarray([[[0.3, 0.2, 0.25, 0.15, 0.1] for _ in range(2)] for _ in range(2)])
    y = np.asarray([[[0.3, 0.19, 0.23, 0.15, 0.1] for _ in range(2)] for _ in range(2)])

    p_values = get_p_at_top_k(x, y, 3)

    np.testing.assert_array_equal(p_values, [[0.735 for _ in range(2)] for _ in range(2)])


def get_top_p_union(x, y, p=DEFAULT_P):
    x_indices = np.argsort(x, axis=-1)[:, :, ::-1]
    y_indices = np.argsort(y, axis=-1)[:, :, ::-1]

    x_results = list()
    y_results = list()
    for i, (x_seq, y_seq) in enumerate(zip(x, y)):
        x_seq_results = list()
        y_seq_results = list()
        for j, (x_token, y_token) in enumerate(zip(x_seq, y_seq)):
            _, top_x_indices = get_top_p(x_token, p)
            _, top_y_indices = get_top_p(y_token, p)
            top_indices = list(set(top_x_indices).union(set(top_y_indices)))
            top_x = x_token[top_indices]
            top_y = y_token[top_indices]

            x_seq_results.append(top_x)
            y_seq_results.append(top_y)

        x_results.append(x_seq_results)
        y_results.append(y_seq_results)

    return x_results, y_results


def assert_top_p_allclose(x, y, atol, rtol, p=DEFAULT_P):
    x, y = get_top_p_union(x, y, p)

    for i, (x_seq, y_seq) in enumerate(zip(x, y)):
        for j, (x_token, y_token) in enumerate(zip(x_seq, y_seq)):
            np.testing.assert_allclose(x_token, y_token, atol=atol, rtol=rtol, equal_nan=False)


def assert_top_k_allclose(x, y, atol, rtol, k=DEFAULT_K):
    x, y = get_top_k_union(x, y, k)
    for i, (x_seq, y_seq) in enumerate(zip(x, y)):
        for j, (x_token, y_token) in enumerate(zip(x_seq, y_seq)):
            np.testing.assert_allclose(x_token, y_token, atol=atol, rtol=rtol, equal_nan=False)


def assert_top_p_and_top_k_allclose(
    x, y, atol_high, rtol_high, atol_low, rtol_low, threshold=1e-3, k=DEFAULT_K, p=DEFAULT_P
):
    """
    the probability distribution has a long tail therefore it is not feasible
    to have good single atol and rtol to both pass large values and cover small
    values at the same time. So breaking down values into high and low values
    """
    x_indices = np.argsort(x, axis=-1)[:, :, ::-1]
    y_indices = np.argsort(y, axis=-1)[:, :, ::-1]

    top_x_flat = list()
    top_y_flat = list()
    for i, (x_seq, y_seq) in enumerate(zip(x, y)):
        for j, (x_token, y_token) in enumerate(zip(x_seq, y_seq)):
            top_x_indices = list()
            x_total = 0.0

            # top p / k in x
            for idx in x_indices[i][j]:
                x_total += x_token[idx]
                top_x_indices.append(idx)

                if x_total > p and len(top_x_indices) >= k:
                    break

            top_y_indices = list()
            y_total = 0.0
            # top p / k in y
            for idx in y_indices[i][j]:
                y_total += y_token[idx]
                top_y_indices.append(idx)

                if y_total > p and len(top_y_indices) >= k:
                    break

            top_indices = list(set(top_x_indices).union(set(top_y_indices)))
            top_x = x_token[top_indices]
            top_y = y_token[top_indices]

            top_x_flat.extend(top_x)
            top_y_flat.extend(top_y)

    top_x_flat = np.asarray(top_x_flat)
    top_y_flat = np.asarray(top_y_flat)

    high_indices = np.where(top_x_flat > threshold)
    top_x_flat_high = top_x_flat[high_indices]
    top_y_flat_high = top_y_flat[high_indices]
    low_indices = np.where(top_x_flat <= threshold)
    top_x_flat_low = top_x_flat[low_indices]
    top_y_flat_low = top_y_flat[low_indices]

    print("checkping higher end:")
    print(
        "smallest prob in top k and top p:", min(np.min(top_x_flat_high), np.min(top_y_flat_high))
    )
    print("number of values:", len(top_x_flat_high) / x.shape[0] / x.shape[1])

    rdiff = relative_difference(top_x_flat_high, top_y_flat_high)
    print("max top p/k difference", np.max(np.abs(top_x_flat_high - top_y_flat_high)))
    print("mean top p/k difference", np.mean(np.abs(top_x_flat_high - top_y_flat_high)))
    print("max top p/k relative difference", np.max(rdiff))
    print("mean top p/k relative difference", np.mean(rdiff))

    np.testing.assert_allclose(
        top_x_flat_high, top_y_flat_high, atol=atol_high, rtol=rtol_high, equal_nan=False
    )

    print("checkping lower end:")
    print("smallest prob in top k and top p:", min(np.min(top_x_flat_low), np.min(top_y_flat_low)))

    rdiff = relative_difference(top_x_flat_low, top_y_flat_low)
    print("max top p/k difference", np.max(np.abs(top_x_flat_low - top_y_flat_low)))
    print("mean top p/k difference", np.mean(np.abs(top_x_flat_low - top_y_flat_low)))
    print("max top p/k relative difference", np.max(rdiff))
    print("mean top p/k relative difference", np.mean(rdiff))

    np.testing.assert_allclose(
        top_x_flat_low, top_y_flat_low, atol=atol_low, rtol=rtol_low, equal_nan=False
    )


def generate_ranges(low, high):
    ranges = [low]
    while ranges[-1] < high:
        step = ranges[-1]
        for i in range(9):
            ranges.append(round(ranges[-1] + step, 15))
    return ranges


def generate_tolerance_map(fuji_model_name, llama_model_name, p=DEFAULT_P):
    fuji_probs = np.load(f"{fuji_model_name}_probs.npy")
    llama_probs = np.load(f"{llama_model_name}_probs.npy")

    threshold = 1e-3
    # threshold = 1e-4

    # generate for high end
    if threshold == 1e-3:
        atols = generate_ranges(1e-5, 5e-4)
        rtols = generate_ranges(1e-3, 1e-2)
    else:
        atols = generate_ranges(1e-6, 1e-4)
        rtols = generate_ranges(1e-3, 1e-2)

    high_end_map = dict()
    high_end_map["atols \ rtols"] = atols

    for rtol in rtols:
        high_end_map[rtol] = list()
        for atol in atols:
            try:
                assert_top_p_and_top_k_allclose(
                    llama_probs,
                    fuji_probs,
                    atol,
                    rtol,
                    1,
                    1,
                    threshold=threshold,
                    p=p,
                )
                high_end_map[rtol].append(" :heavy_check_mark: ")
            except AssertionError:
                high_end_map[rtol].append(" :x: ")

    high_end_df = pd.DataFrame.from_dict(high_end_map)
    high_end_md = high_end_df.to_markdown(index=False)
    print(high_end_md)
    with open("high_end_map.md", "w") as f:
        f.write(high_end_md)

    # generate for lower end
    if threshold == 1e-3:
        atols = generate_ranges(1e-10, 1e-5)
        rtols = generate_ranges(1e-3, 2e-2)
    else:
        atols = generate_ranges(1e-8, 1e-7)
        rtols = generate_ranges(1e-3, 2e-2)

    low_end_map = dict()
    low_end_map["atols \ rtols"] = atols

    for rtol in rtols:
        low_end_map[rtol] = list()
        for atol in atols:
            try:
                assert_top_p_and_top_k_allclose(
                    llama_probs,
                    fuji_probs,
                    1,
                    1,
                    atol,
                    rtol,
                    threshold=threshold,
                    p=p,
                )
                low_end_map[rtol].append(" :heavy_check_mark: ")
            except AssertionError:
                low_end_map[rtol].append(" :x: ")

    low_end_df = pd.DataFrame.from_dict(low_end_map)
    low_end_md = low_end_df.to_markdown(index=False)
    print(low_end_md)
    with open("low_end_map.md", "w") as f:
        f.write(low_end_md)


def validate_probs(fuji_model_name, llama_model_name, p=DEFAULT_P, threshold=None):
    fuji_probs = np.load(f"{fuji_model_name}_probs.npy")
    llama_probs = np.load(f"{llama_model_name}_probs.npy")

    top_k_smallest = get_top_k_smallest(llama_probs, fuji_probs)
    top_k_smallest = min(min(seq_values) for seq_values in top_k_smallest)
    if threshold is None:
        # Selecting a threshold per discussion with Hah
        threshold_min, threshold_max = 1e-5, 1e-4
        threshold = min(top_k_smallest, threshold_max)
        threshold = max(threshold, threshold_min)

    print("threshold:", threshold)

    if threshold == 1e-3:
        atol_high = 1e-5
        rtol_high = 1e-2
        atol_low = 1e-9
        rtol_low = 2e-2
    elif threshold == 1e-4:
        atol_high = 1e-6
        rtol_high = 2e-2
        atol_low = 1e-9
        rtol_low = 2e-2
    else:
        raise Exception(f"atol/rtol not set for unknown threshold: {threshold}")

    # (0.00065897405 - 3.4e-5) / 0.19473135
    # assert_top_k_allclose(llama_probs, fuji_probs, atol_high, rtol_high)
    # assert_top_p_allclose(llama_probs, fuji_probs, atol, rtol, p=0.99)
    assert_top_p_and_top_k_allclose(
        llama_probs,
        fuji_probs,
        atol_high,
        rtol_high,
        atol_low,
        rtol_low,
        threshold=threshold,
        p=p,
    )


def sample_analysis(fuji_model_name, llama_model_name):
    fuji_probs = np.load(f"{fuji_model_name}_probs.npy")
    llama_probs = np.load(f"{llama_model_name}_probs.npy")
    # skip bos_token which has the same prob value for all samples
    fuji_probs = fuji_probs[::, 1::, ::]
    llama_probs = llama_probs[::, 1::, ::]

    results = {
        "smallest in top k": list(),
        "smallest in top p (0.95)": list(),
        "smallest in top p (0.90)": list(),
        "p at top k": list(),
    }
    print("Smallest values in top k per sample:")
    top_k_smallest = get_top_k_smallest(llama_probs, fuji_probs)
    for sample in top_k_smallest:
        results["smallest in top k"].append(min(sample))

    print("Smallest values in top p (0.95) per sample:")
    top_p_smallest = get_top_p_smallest(llama_probs, fuji_probs, p=0.95)
    for sample in top_p_smallest:
        results["smallest in top p (0.95)"].append(min(sample))

    print("Smallest values in top p (0.90) per sample:")
    top_p_smallest = get_top_p_smallest(llama_probs, fuji_probs, p=0.90)
    for sample in top_p_smallest:
        results["smallest in top p (0.90)"].append(min(sample))

    print("p values in top k per sample:")
    p_values = get_p_at_top_k(llama_probs, fuji_probs)
    for sample in p_values:
        results["p at top k"].append(np.mean(sample))

    df = pd.DataFrame.from_dict(results)
    print(df.to_markdown())


texts = [
    "How are you doing?",
    "who is the president of the US now?",
    "The USA is in which continent?",
    "California is a state in",
    "Can you tell me something about California state?\n",
]


def run_gpu_checkpoint_tests(load_true_model=False, reverse=True):
    from axlearn_inference import validate_conversion

    print("Converting and validating Axlearn GPU to HF on 7B...")
    validate_conversion(
        "fuji-7B-v2",
        "Llama-2-7b-hf",
        load_true_model=load_true_model,
        reverse=reverse,
        texts=texts,
        fuji_model_path="/fsx/czhenguo/Projects/fruitstand/runs/artifacts/axlearn_venv/baselines/10976/axlearn_out/checkpoints/step_00034000",
        trn_checkpoint=False,
    )
    validate_probs("fuji-7B-v2", "Llama-2-7b-hf", threshold=1e-3)

    print("Converting and validating Axlearn GPU to HF on 70B...")
    validate_conversion(
        "fuji-70B-v2",
        "Llama-2-70b-hf",
        load_true_model=load_true_model,
        reverse=reverse,
        texts=texts,
        fuji_model_path="/fsx/czhenguo/Projects/fruitstand/runs/artifacts/axlearn_venv/baselines/10985/axlearn_out/checkpoints/step_00035000",
        use_gqa=True,
        trn_checkpoint=False,
    )
    validate_probs("fuji-70B-v2", "Llama-2-70b-hf", threshold=1e-3)


def run_trn_checkpoint_tests(load_true_model=False, reverse=True):
    from axlearn_inference import validate_conversion

    print(
        "Make sure StackedTransformerLayer and GroupedQKVLinear are used when converting TRN ckpt!"
    )
    print("Converting and validating Axlearn TRN to HF on 7B...")
    validate_conversion(
        "fuji-7B-v2",
        "Llama-2-7b-hf",
        load_true_model=load_true_model,
        reverse=reverse,
        texts=texts,
        trn_checkpoint=True,
    )
    validate_probs("fuji-7B-v2", "Llama-2-7b-hf", threshold=1e-3)

    print("Converting and validating Axlearn TRN to HF on 70B...")
    validate_conversion(
        "fuji-70B-v2",
        "Llama-2-70b-hf",
        load_true_model=load_true_model,
        reverse=reverse,
        texts=texts,
        fuji_model_path="/fsx/czhenguo/Projects/fruitstand/runs/artifacts/241230232345/axlearn_out/checkpoints/step_00000002",
        use_gqa=True,
        trn_checkpoint=True,
    )
    validate_probs("fuji-70B-v2", "Llama-2-70b-hf", threshold=1e-4)


def run_gpu_checkpoint_round_trip_tests():
    from axlearn_inference import extend_texts, run_forward_pass

    batch_texts = extend_texts(texts, 8)

    original_probs = run_forward_pass(
        "fuji-7B-v2",
        batch_texts,
        "/fsx/czhenguo/Projects/fruitstand/runs/artifacts/axlearn_venv/baselines/10976/axlearn_out/checkpoints/step_00034000",
    )
    round_trip_probs = run_forward_pass(
        "fuji-7B-v2",
        batch_texts,
        "/fsx/czhenguo/Projects/fruitstand/runs/artifacts/transformers_to_axlearn/round_trip/step_00022794",
    )

    np.testing.assert_allclose(
        original_probs, round_trip_probs, atol=0.0, rtol=0.0, equal_nan=False
    )


if __name__ == "__main__":
    validate_probs("fuji-7B-v2", "Llama-2-7b-hf", threshold=1e-3)
    # generate_tolerance_map("fuji-7B-v2", "Llama-2-7b-hf")
    # sample_analysis("fuji-7B-v2", "Llama-2-7b-hf")
    # validate_probs("fuji-70B-v2", "Llama-2-70b-hf", threshold=1e-4)
    # run_gpu_checkpoint_tests(load_true_model=True)
    # run_trn_checkpoint_tests(load_true_model=True)
    # run_gpu_checkpoint_tests(reverse=False)
    # run_trn_checkpoint_tests(reverse=False)
    # run_gpu_checkpoint_round_trip_tests()
    # test_top_p()
    # test_top_k()
    # test_get_top_k_smallest()
    # test_get_top_p_smallest()
    # test_get_p_at_top_k()
