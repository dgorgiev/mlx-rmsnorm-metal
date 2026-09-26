import mlx.core as mx


_PARALLEL_SOURCE = r"""
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint thread_count = threads_per_threadgroup.x;
    uint D = x_shape[x_ndim - 1];
    uint row_offset = row * D;

    float partial = 0.0f;
    for (uint j = tid; j < D; j += thread_count) {
        float value = static_cast<float>(x[row_offset + j]);
        partial += value * value;
    }

    threadgroup float scratch[512];
    scratch[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = thread_count >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_rms = metal::rsqrt(
        scratch[0] / static_cast<float>(D) + eps[0]);

    for (uint j = tid; j < D; j += thread_count) {
        float value = static_cast<float>(x[row_offset + j]);
        float scale = static_cast<float>(weight[j]);
        out[row_offset + j] = static_cast<T>(value * inv_rms * scale);
    }
"""


_parallel_kernel = mx.fast.metal_kernel(
    name="rms_norm_parallel",
    input_names=["x", "weight", "eps"],
    output_names=["out"],
    source=_PARALLEL_SOURCE,
)


def rms_norm_parallel(x, weight, eps=1e-5, threadgroup_size=256):
    """RMSNorm with one threadgroup and a tree reduction per row."""
    _validate_inputs(x, weight, threadgroup_size)
    rows = x.size // x.shape[-1]
    eps_array = mx.array([eps], dtype=mx.float32)
    return _parallel_kernel(
        inputs=[x, weight, eps_array],
        template=[("T", x.dtype)],
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
        grid=(rows * threadgroup_size, 1, 1),
        threadgroup=(threadgroup_size, 1, 1),
    )[0]


def _validate_inputs(x, weight, threadgroup_size):
    if x.ndim == 0:
        raise ValueError("x must have at least one dimension")
    if weight.ndim != 1 or weight.shape[0] != x.shape[-1]:
        raise ValueError("weight must be one-dimensional and match x.shape[-1]")
    if x.dtype != weight.dtype:
        raise ValueError("x and weight must have the same dtype")
    if threadgroup_size not in (32, 64, 128, 256, 512):
        raise ValueError("threadgroup_size must be 32, 64, 128, 256, or 512")
