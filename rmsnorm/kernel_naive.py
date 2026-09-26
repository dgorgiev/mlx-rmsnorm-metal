import mlx.core as mx


_NAIVE_SOURCE = r"""
    uint row = thread_position_in_grid.x;
    uint D = x_shape[x_ndim - 1];
    uint row_offset = row * D;

    float sum = 0.0f;
    for (uint j = 0; j < D; ++j) {
        float value = static_cast<float>(x[row_offset + j]);
        sum += value * value;
    }

    float inv_rms = metal::rsqrt(sum / static_cast<float>(D) + eps[0]);
    for (uint j = 0; j < D; ++j) {
        float value = static_cast<float>(x[row_offset + j]);
        float scale = static_cast<float>(weight[j]);
        out[row_offset + j] = static_cast<T>(value * inv_rms * scale);
    }
"""


_naive_kernel = mx.fast.metal_kernel(
    name="rms_norm_naive",
    input_names=["x", "weight", "eps"],
    output_names=["out"],
    source=_NAIVE_SOURCE,
)


def rms_norm_naive(x, weight, eps=1e-5):
    """RMSNorm with one GPU thread processing each complete row."""
    _validate_inputs(x, weight)
    rows = x.size // x.shape[-1]
    eps_array = mx.array([eps], dtype=mx.float32)
    return _naive_kernel(
        inputs=[x, weight, eps_array],
        template=[("T", x.dtype)],
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
        grid=(rows, 1, 1),
        threadgroup=(min(256, rows), 1, 1),
    )[0]


def _validate_inputs(x, weight):
    if x.ndim == 0:
        raise ValueError("x must have at least one dimension")
    if weight.ndim != 1 or weight.shape[0] != x.shape[-1]:
        raise ValueError("weight must be one-dimensional and match x.shape[-1]")
    if x.dtype != weight.dtype:
        raise ValueError("x and weight must have the same dtype")
