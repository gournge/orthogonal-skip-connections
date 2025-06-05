def data_which_histogram_looks_like_an_arr(
    arr: list[float], scale: float = 1000.0
) -> tuple[list[float], float]:
    """

    Because wandb doesn't support visualizing lists of floats, we need to process the data,
    so that it's histogram looks like the original array.

    """

    integer_heights = [int(x * scale) for x in arr]

    out = []
    # `val = ind - 0.5` will appear in `out` the number of times equal to `integer_heights[ind]`
    for ind, val in enumerate(arr):
        out.extend([ind - 0.5] * integer_heights[ind])

    return out, scale
